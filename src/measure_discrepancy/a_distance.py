
from collections import defaultdict
from collections import namedtuple
from typing import List, Optional
from tqdm import tqdm
import h5py
import numpy as np

from sklearn.linear_model import SGDClassifier
from datasets.constant import UD_HEAD_LABELS
from measure_discrepancy.utils import compute_depth

LOIs = ["uas", "un", "all"] + UD_HEAD_LABELS


class HDF5Dataset:
    DATA_FIELDS = ["token_ids", "tokens", "length", "pos_label_ids",
                   "head_ids", "arc_label_ids", "hidden_state"]

    def __init__(
            self,
            filepaths,
            langs,
            layer_index=12,
            control_size=True,
            label2id=None,
            dp_labels=UD_HEAD_LABELS,
    ):
        super(HDF5Dataset, self).__init__()
        if label2id is None:
            label2id = {l: v for (v, l) in enumerate(dp_labels)}
        self.langs = langs
        self.filepaths = filepaths
        self.layer_index = layer_index
        self.control_size = control_size
        # self.LOI = LOI
        self.label2id = label2id
        self.observation_class = self.get_observation_class(self.DATA_FIELDS)
        self.src_observations = self.load_dataset_group(filepaths[0])
        self.tgt_observations = self.load_dataset_group(filepaths[1])

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def prepare_dataset(self, loi, size_per_lang=None):
        label_diffs = []
        print("Loading Diffs", self.langs[0], self.filepaths[0])
        label_diffs.append(self.get_label_diffs(self.langs[0], self.src_observations, loi=loi))
        print("Loading Diffs", self.langs[1], self.filepaths[1])
        label_diffs.append(self.get_label_diffs(self.langs[1], self.tgt_observations, loi=loi))

        if (loi is None) or (loi == "all") or (loi == "un") or (loi == "uas"):
            if self.control_size:
                if size_per_lang is None:
                    size_per_lang = [np.min([len(label_diffs[i]["labels"]) for i in range(len(label_diffs))])] * 2
                else:
                    size_per_lang = size_per_lang
            else:
                size_per_lang = [len(label_diff["labels"]) for label_diff in label_diffs]

            for i in range(len(self.langs)):
                print(f"Number of [ {loi} ] in {self.langs[i]} dataset: {size_per_lang[i]}")

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

        else:
            label_id = self.label2id[loi]
            if np.min([np.count_nonzero(np.array(
                    label_diffs[i]["labels"]) == label_id) for i in range(len(label_diffs))]) < 30:
                return [np.count_nonzero(
                    np.array(label_diffs[i]["labels"]) == label_id) for i in range(len(label_diffs))]
            if self.control_size:
                if size_per_lang is None:
                    size_per_lang = [np.min([np.count_nonzero(np.array(label_diffs[i]["labels"]) == label_id) for i in
                                             range(len(label_diffs))])] * 2
                else:
                    size_per_lang = size_per_lang
            else:
                size_per_lang = [np.count_nonzero(np.array(label_diffs[i]["labels"]) == label_id) for i in
                                 range(len(label_diffs))]

            assert np.count_nonzero(np.array(label_diffs[0]["labels"]) == label_id) == len(label_diffs[0]["labels"])
            assert np.count_nonzero(np.array(label_diffs[1]["labels"]) == label_id) == len(label_diffs[1]["labels"])

            for i in range(len(self.langs)):
                print(f"Number of [ {loi} ] in {self.langs[i]} dataset: {size_per_lang[i]}")

            outputs_train = defaultdict(list)
            outputs_test = defaultdict(list)

            for i in range(len(label_diffs)):
                mask = np.array(label_diffs[i]["labels"]) == label_id

                idx = np.random.choice(
                    np.arange(len(label_diffs[i]["labels"]))[mask],
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

    def get_label_diffs(self, lang, observations, loi):
        outputs = defaultdict(list)

        for observation in tqdm(observations, desc='[computing labels & diffs]'):
            labels = []
            diffs = []
            if loi is None or loi == "uas":
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        elif observation.head_ids[j] == i:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        else:
                            label = 0
                            i_depth = compute_depth(idx=i, head_ids=observation.head_ids)
                            j_depth = compute_depth(idx=j, head_ids=observation.head_ids)
                            diff = np.sign(i_depth - j_depth) * (
                                        observation.hidden_state[j] - observation.hidden_state[i])
                        labels.append(label)
                        diffs.append(diff)

            elif loi == "all":
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

            elif loi == "un":
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j or observation.head_ids[j] == i:
                            continue
                        else:
                            label = 0
                            i_depth = compute_depth(idx=i, head_ids=observation.head_ids)
                            j_depth = compute_depth(idx=j, head_ids=observation.head_ids)
                            diff = np.sign(i_depth - j_depth) * (
                                    observation.hidden_state[j] - observation.hidden_state[i])
                        labels.append(label)
                        diffs.append(diff)
            else:
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j and observation.arc_label_ids[i] == self.label2id[loi]:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        elif observation.head_ids[j] == i and observation.arc_label_ids[j] == self.label2id[loi]:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        else:
                            continue
                        labels.append(label)
                        diffs.append(diff)

            to_add = {
                "labels": labels,
                "diffs": diffs,
                # "rels": rels,
                "lang": [lang] * len(labels),
            }

            for target in to_add:
                outputs[target] += list(to_add[target])

        return outputs


def compute_A_distance(
        filepaths,
        langs,
        layer_index=12,
        LOI=None,
        alpha_list=np.logspace(-7, 0, num=8, base=10),
        verbose=True,
):
    dataset = HDF5Dataset(filepaths=filepaths, langs=langs, layer_index=layer_index)
    data_train, data_test = dataset.prepare_dataset(loi=LOI)
    if isinstance(data_train, int):
        print(">>> Not enough samples. \n>>> Can't compute a-distance:")
        print(f"Number of [ {LOI} ] in {langs[0]}: {data_train} \nNumber of [ {LOI} ] in {langs[1]}: {data_test}")
        return [data_train, data_test]

    if verbose:
        print(">>> Preparing train and test dataset...")

    train_X = np.vstack((data_train[f"diffs_{langs[0]}"], data_train[f"diffs_{langs[1]}"]))
    train_Y = np.hstack((np.ones(len(data_train[f"labels_{langs[0]}"]), dtype=int),
                         -np.ones(len(data_train[f"labels_{langs[1]}"]), dtype=int)))

    test_X = np.vstack((data_test[f"diffs_{langs[0]}"], data_test[f"diffs_{langs[1]}"]))
    test_Y = np.hstack((np.ones(len(data_test[f"labels_{langs[0]}"]), dtype=int),
                        -np.ones(len(data_test[f"labels_{langs[1]}"]), dtype=int)))

    if verbose:
        print(">>> Computing A-Distance...")

    best_risk = 1.0
    for alpha in alpha_list:
        clf = SGDClassifier(
            loss="modified_huber",
            alpha=alpha,
            max_iter=10000,
            shuffle=True,
            verbose=False,
            random_state=25,
            early_stopping=True
        )
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ alpha = %f ] train risk: %f  test risk: %f' % (alpha, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    a_distance = 2 * (1. - 2 * best_risk)

    if verbose:
        print("A-Distance: %f" % a_distance)
    return a_distance


def compute_a_dist_with_dataset(
        dataset, loi, langs,
        alpha_list=np.logspace(-7, 0, num=8, base=10), verbose=True,
):
    data_train, data_test = dataset.prepare_dataset(loi=loi)
    if isinstance(data_train, int):
        print(">>> Not enough samples. \n>>> Can't compute a-distance:")
        print(f"Number of [ {loi} ] in {langs[0]}: {data_train} \nNumber of [ {loi} ] in {langs[1]}: {data_test}")
        return [data_train, data_test]
    if verbose:
        print(">>> Preparing train and test dataset...")

    train_X = np.vstack((data_train[f"diffs_{langs[0]}"], data_train[f"diffs_{langs[1]}"]))
    train_Y = np.hstack((np.ones(len(data_train[f"labels_{langs[0]}"]), dtype=int),
                         -np.ones(len(data_train[f"labels_{langs[1]}"]), dtype=int)))

    test_X = np.vstack((data_test[f"diffs_{langs[0]}"], data_test[f"diffs_{langs[1]}"]))
    test_Y = np.hstack((np.ones(len(data_test[f"labels_{langs[0]}"]), dtype=int),
                        -np.ones(len(data_test[f"labels_{langs[1]}"]), dtype=int)))

    if verbose:
        print(">>> Computing A-Distance...")

    best_risk = 1.0
    for alpha in alpha_list:
        clf = SGDClassifier(
            loss="modified_huber",
            alpha=alpha,
            max_iter=10000,
            shuffle=True,
            verbose=False,
            random_state=25,
            early_stopping=True
        )
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ alpha = %f ] train risk: %f  test risk: %f' % (alpha, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    a_distance = 2 * (1. - 2 * best_risk)

    if verbose:
        print("A-Distance: %f" % a_distance)
    return a_distance
