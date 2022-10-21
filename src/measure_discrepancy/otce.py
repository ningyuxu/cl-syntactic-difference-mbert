
from collections import defaultdict
from collections import namedtuple
from typing import List, Optional
from tqdm import tqdm

import math

import h5py
import numpy as np

import torch

from datasets.constant import UD_HEAD_LABELS
from measure_discrepancy.utils import compute_depth

import ot
import geomloss


LOIs = [
    #"uas",
    #"las",
    "all"
]


class HDF5Dataset:
    DATA_FIELDS = ["token_ids", "tokens", "length", "pos_label_ids",
                   "head_ids", "arc_label_ids", "hidden_state"]

    def __init__(
            self,
            filepaths,
            langs,
            layer_index=12,
            # batch_size=20,
            control_size=True,
            LOI=None,
            label2id=None,
            dp_labels=UD_HEAD_LABELS,
    ):
        super(HDF5Dataset, self).__init__()
        if label2id is None:
            label2id = {l: v for (v, l) in enumerate(dp_labels)}
        self.langs = langs
        self.filepaths = filepaths
        self.layer_index = layer_index
        # self.batch_size = batch_size
        self.observation_class = self.get_observation_class(self.DATA_FIELDS)
        self.control_size = control_size
        self.LOI = LOI
        self.label2id = label2id

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def prepare_dataset(self, size_per_lang=None):
        label_diffs = []
        for i in range(len(self.filepaths)):
            print("Loading", self.langs[i], self.filepaths[i])
            observations = self.load_dataset_group(self.filepaths[i])
            label_diffs.append(self.get_label_diffs(self.langs[i], observations))

        if self.control_size:
            if size_per_lang is None:
                size_per_lang = [np.min([len(label_diffs[i]["labels"]) for i in range(len(label_diffs))])] * 2
            else:
                size_per_lang = size_per_lang
        else:
            size_per_lang = [len(label_diff["labels"]) for label_diff in label_diffs]

        if (self.LOI is None) or (self.LOI == "las") or (self.LOI == "all") or (self.LOI == "uas"):
            for i in range(len(self.langs)):
                print(f"Number of samples in {self.langs[i]} dataset: {size_per_lang[i]}")
        else:
            for i in range(len(self.langs)):
                print(f"Number of [ {self.LOI} ] in {self.langs[i]} dataset: {size_per_lang[i]}")

        outputs = defaultdict(list)
        for i in range(len(label_diffs)):
            if len(label_diffs[i]["labels"]) < size_per_lang[i]:
                print(f"Number of [ {self.LOI} ] in {self.langs[i]} dataset less than {size_per_lang[i]}.")
                print(">>> Replace set to True.")
                all_idx = np.arange(len(label_diffs[i]["labels"]))
                add_idx = np.random.choice(
                    np.arange(len(label_diffs[i]["labels"])),
                    size=size_per_lang[i]-len(label_diffs[i]["labels"]),
                    replace=True
                )
                idx = np.concatenate((all_idx, add_idx))
            else:
                idx = np.random.choice(
                    np.arange(len(label_diffs[i]["labels"])),
                    size=size_per_lang[i],
                    replace=False
                )
            to_add = {
                f"labels_{self.langs[i]}": np.array(label_diffs[i]["labels"])[idx].tolist(),
                f"diffs_{self.langs[i]}": np.array(label_diffs[i]["diffs"])[idx].tolist(),
            }
            for target in to_add:
                outputs[target] += list(to_add[target])
        return outputs

    def get_observation_class(self, fieldnames):
        return namedtuple('Observation', fieldnames, defaults=(None,) * len(fieldnames))

    def load_dataset_group(self, filepath):
        observations = []
        data = dict()
        hdf5_file = h5py.File(filepath, 'r')
        indices = list(hdf5_file.keys())
        for idx in tqdm(sorted([int(x) for x in indices]), desc='[loading observations]'):
            to_del = 0
            length = hdf5_file[str(idx)]["length"][()]
            for key in self.DATA_FIELDS:
                if key == "hidden_state":
                    data[key] = hdf5_file[str(idx)][key][()][self.layer_index][:length]
                elif key == "length":
                    data[key] = hdf5_file[str(idx)][key][()]
                else:
                    data[key] = hdf5_file[str(idx)][key][()][:length]
            observation = self.observation_class(**data)
            for head in observation.head_ids:
                if head > observation.length - 1:
                    to_del = 1
            if to_del:
                continue
            else:
                observations.append(observation)
        return observations

    def get_label_diffs(self, lang, observations):
        outputs = defaultdict(list)
        for observation in tqdm(observations, desc='[computing labels & diffs]'):
            labels = []
            diffs = []
            if (self.LOI is None) or (self.LOI == "las") or (self.LOI == "uas"):
                count_non = 0
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        elif observation.head_ids[j] == i:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        elif count_non < observation.length - 2:
                            label = 0
                            i_depth = compute_depth(idx=i, head_ids=observation.head_ids)
                            j_depth = compute_depth(idx=j, head_ids=observation.head_ids)
                            diff = np.sign(i_depth - j_depth) * (
                                    observation.hidden_state[j] - observation.hidden_state[i])
                            count_non += 1
                        else:
                            continue
                        labels.append(label)
                        diffs.append(diff)

            elif self.LOI == "all":
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        elif observation.head_ids[j] == i:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        else:
                            continue
                        labels.append(label)
                        diffs.append(diff)

            else:
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j and observation.arc_label_ids[i] == self.label2id[self.LOI]:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        elif observation.head_ids[j] == i and observation.arc_label_ids[j] == self.label2id[self.LOI]:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        else:
                            continue
                        labels.append(label)
                        diffs.append(diff)

            to_add = {
                "labels": labels,
                "diffs": diffs,
                "lang": [lang] * len(labels),
            }

            for target in to_add:
                outputs[target] += list(to_add[target])

        return outputs


def prepare_dataset(
        filepaths,
        langs,
        layer_index=12,
        loi=None,
        size_per_lang=None,
        kind="uas",
        verbose=True,
):
    data = HDF5Dataset(
        filepaths=filepaths,
        langs=langs,
        layer_index=layer_index,
        LOI=loi
    ).prepare_dataset(size_per_lang=size_per_lang)

    if verbose:
        print(">>> Preparing dataset...")

    src_x = torch.tensor(np.array(data[f"diffs_{langs[0]}"]), dtype=torch.float)
    tgt_x = torch.tensor(np.array(data[f"diffs_{langs[1]}"]), dtype=torch.float)

    if kind == "las":
        src_y = np.array(data[f"labels_{langs[0]}"])[:, np.newaxis]
        tgt_y = np.array(data[f"labels_{langs[1]}"])[:, np.newaxis]
    else:
        src_y = (np.array(data[f"labels_{langs[0]}"]) != 0).astype("int")[:, np.newaxis]
        tgt_y = (np.array(data[f"labels_{langs[1]}"]) != 0).astype("int")[:, np.newaxis]

    return src_x, tgt_x, src_y, tgt_y


def compute_coupling(x_src, x_tgt):
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(x_src, x_tgt)
    P = ot.emd(torch.tensor(ot.unif(x_src.shape[0])), torch.tensor(ot.unif(x_tgt.shape[0])), C, numItermax=10000000)
    W = np.sum(P.cpu().numpy() * C.cpu().numpy())

    return P, W


def compute_CE(P, y_src, y_tgt):
    src_label_set = set(sorted(list(y_src.flatten())))
    tgt_label_set = set(sorted(list(y_tgt.flatten())))

    # joint distribution of source and target label
    P_src_tgt = np.zeros((np.max(y_src)+1, np.max(y_tgt)+1))

    for y1 in src_label_set:
        y1_idx = np.where(y_src == y1)
        for y2 in tgt_label_set:
            y2_idx = np.where(y_tgt == y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

            P_src_tgt[y1, y2] = np.sum(P[RR, CC])

    # marginal distribution of source label
    P_src = np.sum(P_src_tgt, axis=1)

    ce = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tgt_label_set:
            if P_src_tgt[y1, y2] != 0:
                ce += -(P_src_tgt[y1, y2] * math.log(P_src_tgt[y1, y2] / P_y1))
    return ce


def compute_w_ce(filepaths, langs, device, layer_index=12,
                 loi=None, size_per_lang=None, kind="uas",
                 verbose=True):
    if size_per_lang is None:
        size_per_lang = [20000, 20000]

    src_x, tgt_x, src_y, tgt_y = prepare_dataset(
        filepaths=filepaths,
        langs=langs,
        layer_index=layer_index,
        loi=loi,
        size_per_lang=size_per_lang,
        kind=kind,
        verbose=verbose
    )
    P, W = compute_coupling(src_x.to(device), tgt_x.to(device))
    ce = compute_CE(P.cpu().numpy(), src_y, tgt_y)

    if verbose:
        print(f"Disparity between {langs[0]} and {langs[1]}:")
        print("Wasserstein distance: %.4f, Conditonal Entropy: %.4f" % (W, ce))
    return W, ce
