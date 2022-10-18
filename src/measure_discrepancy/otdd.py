
from collections import defaultdict
from collections import namedtuple
from typing import List
from tqdm import tqdm

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from datasets.constant import UD_HEAD_LABELS

from otdd.pytorch.distance import DatasetDistance

from measure_discrepancy.utils import SubsetSampler, compute_depth
from otdd.pytorch.utils import random_index_split


LOIs = [
    # "uas",
    # "las",
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
            loi=None,
            label2id=None,
            dp_labels=UD_HEAD_LABELS,
            size_per_lang=None
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
        self.LOI = loi
        self.label2id = label2id
        self.data = self.prepare_dataset(size_per_lang=size_per_lang)
        self.src_dataset = ObservationIterator(self.data, langs[0], loi=loi)
        self.tgt_dataset = ObservationIterator(self.data, langs[1], loi=loi)

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def prepare_dataset(self, size_per_lang=None, kind="otce"):
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

        if (self.LOI is None) or (self.LOI == "las") or (self.LOI == "uas") or (self.LOI == "all"):
            for i in range(len(self.langs)):
                print(f"Number of samples in {self.langs[i]} dataset: {size_per_lang[i]}")
        else:
            for i in range(len(self.langs)):
                print(f"Number of [ {self.LOI} ] in {self.langs[i]} dataset: {size_per_lang[i]}")

        if kind == "a-distance":
            outputs_train = defaultdict(list)
            outputs_test = defaultdict(list)

            for i in range(len(label_diffs)):
                idx = np.random.choice(
                    np.arange(len(label_diffs[i]["labels"])),
                    size=size_per_lang[i],
                    replace=False
                )
                half = int(size_per_lang[i] / 2) + 1
                train_idx = idx[:half]
                test_idx = idx[half:]

                to_add_train = {
                    f"labels_{self.langs[i]}": np.array(label_diffs[i]["labels"])[train_idx].tolist(),
                    f"diffs_{self.langs[i]}": np.array(label_diffs[i]["diffs"])[train_idx].tolist(),
                }
                for target in to_add_train:
                    outputs_train[target] += list(to_add_train[target])

                to_add_test = {
                    f"labels_{self.langs[i]}": np.array(label_diffs[i]["labels"])[test_idx].tolist(),
                    f"diffs_{self.langs[i]}": np.array(label_diffs[i]["diffs"])[test_idx].tolist(),
                }
                for target in to_add_test:
                    outputs_test[target] += list(to_add_test[target])

            return outputs_train, outputs_test

        else:
            outputs = defaultdict(list)

            for i in range(len(label_diffs)):
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
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        elif observation.head_ids[j] == i:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        elif count_non < observation.length - 2:
                            label = 0
                            i_depth = compute_depth(idx=i, head_ids=observation.head_ids)
                            j_depth = compute_depth(idx=j, head_ids=observation.head_ids)
                            diff = np.sign(i_depth - j_depth) * (
                                    observation.hidden_state[i] - observation.hidden_state[j])
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
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        elif observation.head_ids[j] == i:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        else:
                            continue
                        labels.append(label)
                        diffs.append(diff)

            else:
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j and observation.arc_label_ids[i] == self.label2id[self.LOI]:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        elif observation.head_ids[j] == i and observation.arc_label_ids[j] == self.label2id[self.LOI]:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
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


class ObservationIterator(TorchDataset):
    def __init__(self, data, lang, labels=None, targets=None, loi=None):
        self.data = data
        self.lang = lang
        self._labels = labels
        self._targets = targets
        self.loi = loi

    def __len__(self):
        return len(self.data[f"labels_{self.lang}"])

    def __getitem__(self, idx):
        x = torch.tensor(self.data[f"diffs_{self.lang}"], dtype=torch.float)[idx]
        if self.loi is None or self.loi == "uas":
            y = torch.LongTensor(np.array(self.data[f"labels_{self.lang}"]) != 0)[idx]
        else:
            y = torch.LongTensor(self.data[f"labels_{self.lang}"])[idx]
        return x, y

    @property
    def targets(self):
        if self._targets is None:
            if self.loi is None or self.loi == "uas":
                self._targets = torch.LongTensor(np.array(self.data[f"labels_{self.lang}"]) != 0)
            else:
                self._targets = torch.LongTensor(self.data[f"labels_{self.lang}"])
        return self._targets

    @property
    def classes(self):
        if self._labels is None:
            if self.loi is None or self.loi == "uas":
                self._labels = [0, 1]
            else:
                self._labels = UD_HEAD_LABELS
        return self._labels


def load_dataset(
        data, valid_size=0.1, splits=None,
        maxsize=None, shuffle=True, batch_size=16,
        num_workers=0, random_seed=25,
):
    if shuffle and random_seed:
        np.random.seed(random_seed)

    fold_idxs = {}
    if splits is None and valid_size == 0:
        fold_idxs['train'] = np.arange(len(data))
    elif splits is None and valid_size > 0:
        train_idx, valid_idx = random_index_split(len(data), 1 - valid_size,
                                                  (maxsize, None))  # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx

    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace=False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)
    fold_loaders = {k: TorchDataLoader(data, sampler=sampler, **dataloader_args)
                    for k, sampler in fold_samplers.items()}
    return fold_loaders


def compute_otdd(
        filepaths,
        langs,
        layer_index=12,
        loi="las",
        size_per_lang=None,
        splits=None,
        num_workers=0,
        valid_size=0,
        maxsize=2000,
        batch_size=16,
        shuffle=True,
        random_seed=25,
        device="cpu",
        maxsamples=5000,
        return_coupling=False,
        verbose=True,
):
    dataset = HDF5Dataset(filepaths=filepaths, langs=langs, layer_index=layer_index,
                          loi=loi, size_per_lang=size_per_lang)

    src = load_dataset(dataset.src_dataset, valid_size=valid_size, splits=splits,
                       maxsize=maxsize, batch_size=batch_size, shuffle=shuffle,
                       random_seed=random_seed, num_workers=num_workers)
    tgt = load_dataset(dataset.tgt_dataset, valid_size=valid_size, splits=splits,
                       maxsize=maxsize, batch_size=batch_size, shuffle=shuffle,
                       random_seed=random_seed, num_workers=num_workers)

    dist = DatasetDistance(src['train'], tgt['train'], inner_ot_method='exact',
                           debiased_loss=True, p=2, entreg=1e-1, device=device)

    if return_coupling:
        d, pi = dist.distance(maxsamples=maxsamples, return_coupling=return_coupling)
        label_dist = (dist.label_distances).cpu().numpy().tolist()
        src_classes = dist.classes1
        tgt_classes = dist.classes2
        print(f"OTDD between [ {langs[0]} ] and [ {langs[1]} ] : {d}")
        return d, pi, label_dist, src_classes, tgt_classes

    else:
        d = dist.distance(maxsamples=maxsamples, return_coupling=return_coupling).cpu().numpy().item()
        label_dist = (dist.label_distances).cpu().numpy().tolist()
        src_classes = dist.classes1
        tgt_classes = dist.classes2
        print(f"OTDD between [ {langs[0]} ] and [ {langs[1]} ] : {d}")
        return d, label_dist, src_classes, tgt_classes
