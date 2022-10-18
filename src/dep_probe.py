import os
import json
import time
import shutil
from tqdm import tqdm

import argparse
from argparse import Namespace

import re
import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from collections import namedtuple
from typing import List, Optional
import h5py

from sklearn.linear_model import SGDClassifier
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
import torch.nn as nn

from parsing_data import get_metric
from datasets.constant import ISO638_TO_LANG, UD_HEAD_LABELS
from measure_discrepancy.utils import compute_depth

from transformers import logging

from datasets.constant import UD_HEAD_LABELS, UD_ISO638_TO_LANG, ISO638_TO_LANG
from dataloader.base import DataLoader
from models.parsing import ParsingModel
from tasks.base import Task
from models.xbert import XBertModel
from experiment.experiment import Experiment
from experiment import experiment
import helper.enumeration as enum
from compute_dist import train_model, analyse_and_path
from compute_dist_pretrained import analyse_and_path as pretrain_analyse_and_path


logging.set_verbosity_error()


REPORT_PATH = "./experiment/probe/"

LOIs = ["all"]


def write_res(res_dict, lang_iso, out_path=REPORT_PATH, model="fine-tuned", layer=12):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if model == "pretrain":
        file_path = os.path.join(out_path, f"pretrain_{lang_iso}_{layer}.json")
    elif model == "random":
        file_path = os.path.join(out_path, f"random_{lang_iso}_{layer}.json")
    else:
        file_path = os.path.join(out_path, f"finetune_{lang_iso}_{layer}.json")
    with open(file_path, mode='a', encoding="utf-8") as f:
        logs = dict()
        for k, v in res_dict.items():
            logs[k] = v
        print(json.dumps(logs), file=f)


class HDF5Dataset:
    DATA_FIELDS = ["token_ids", "tokens", "length", "pos_label_ids",
                   "head_ids", "arc_label_ids", "hidden_state"]

    def __init__(
            self,
            filepaths,
            lang,
            layer_index=12,
            # batch_size=20,
            control_size=True,
            label2id=None,
            dp_labels=UD_HEAD_LABELS,
    ):
        super(HDF5Dataset, self).__init__()
        if label2id is None:
            label2id = {l: v for (v, l) in enumerate(dp_labels)}
        self.lang = lang
        self.filepaths = filepaths
        self.layer_index = layer_index
        # self.batch_size = batch_size
        self.observation_class = self.get_observation_class(self.DATA_FIELDS)
        self.control_size = control_size
        self.label2id = label2id
        self.train_observations = self.load_dataset_group(filepaths[0])
        self.test_observations = self.load_dataset_group(filepaths[-1])
        # self.test_observations = self.load_dataset_group(filepaths[2])

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def prepare_dataset(self, loi):
        train_label_diffs = []
        print("Loading Diffs", self.lang, self.filepaths[0])
        train_label_diffs.append(self.get_label_diffs(self.lang, self.train_observations, loi=loi))
        test_label_diffs = []
        print("Loading Diffs", self.lang, self.filepaths[-1])
        test_label_diffs.append(self.get_label_diffs(self.lang, self.test_observations, loi=loi))

        outputs_train = defaultdict(list)
        outputs_test = defaultdict(list)

        to_drop_label_id = []
        if loi == "all":
            for label_id in range(len(UD_HEAD_LABELS)):
                if np.min([np.count_nonzero(np.array(
                        train_label_diffs[i]["labels"]) == label_id) for i in range(len(train_label_diffs))]) < 5:
                    to_drop_label_id.append(label_id)
        else:
            for label_id in [0, 1]:
                if np.min([np.count_nonzero(np.array(
                        train_label_diffs[i]["labels"]) == label_id) for i in range(len(train_label_diffs))]) < 5:
                    raise AssertionError

        for i in range(len(train_label_diffs)):
            mask_train = [label_id not in to_drop_label_id for label_id in train_label_diffs[i]["labels"]]
            to_add_train = {
                f"labels_{self.lang}": np.array(train_label_diffs[i]["labels"])[mask_train].tolist(),
                f"diffs_{self.lang}": np.array(train_label_diffs[i]["diffs"])[mask_train].tolist(),
            }
            for target in to_add_train:
                outputs_train[target] += list(to_add_train[target])

        for i in range(len(test_label_diffs)):
            mask_test = [label_id not in to_drop_label_id for label_id in test_label_diffs[i]["labels"]]
            to_add_test = {
                f"labels_{self.lang}": np.array(test_label_diffs[i]["labels"])[mask_test].tolist(),
                f"diffs_{self.lang}": np.array(test_label_diffs[i]["diffs"])[mask_test].tolist(),
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
                count_non = 0
                for i in range(1, observation.length - 1):
                    for j in range(1, i):
                        if observation.head_ids[i] == j:
                            label = 1
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        elif observation.head_ids[j] == i:
                            label = 1
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

            elif loi == "all":
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
                        if observation.head_ids[i] == j and observation.arc_label_ids[i] == self.label2id[loi]:
                            label = observation.arc_label_ids[i]
                            diff = observation.hidden_state[i] - observation.hidden_state[j]
                        elif observation.head_ids[j] == i and observation.arc_label_ids[j] == self.label2id[loi]:
                            label = observation.arc_label_ids[j]
                            diff = observation.hidden_state[j] - observation.hidden_state[i]
                        else:
                            label = 0
                            i_depth = compute_depth(idx=i, head_ids=observation.head_ids)
                            j_depth = compute_depth(idx=j, head_ids=observation.head_ids)
                            diff = np.sign(i_depth - j_depth) * (
                                    observation.hidden_state[i] - observation.hidden_state[j])
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


def probing(dataset, loi, lang,
            alpha_list=np.logspace(-9, -2, num=8, base=10),
            verbose=True):
    data_train, data_test = dataset.prepare_dataset(loi=loi)
    train_X = torch.tensor(np.array(data_train[f"diffs_{lang}"]), dtype=torch.float)
    train_Y = np.array(data_train[f"labels_{lang}"])

    test_X = torch.tensor(np.array(data_test[f"diffs_{lang}"]), dtype=torch.float)
    test_Y = np.array(data_test[f"labels_{lang}"])

    accs = []

    for alpha in alpha_list:
        clf = SGDClassifier(
            loss="log",
            alpha=alpha,
            max_iter=10000,
            shuffle=True,
            verbose=False,
            random_state=25,
            early_stopping=True
        )
        clf.fit(train_X, train_Y)
        train_acc = np.mean(clf.predict(train_X) == train_Y)
        test_acc = np.mean(clf.predict(test_X) == test_Y)

        if verbose:
            print('[ alpha = %f ] train acc: %f  test acc: %f' % (alpha, train_acc, test_acc))

        accs.append(test_acc)
    return accs


def get_args(task):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="pretrain", type=str)
    parser.add_argument('--src_lang', default="English", type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser = DataLoader.add_model_specific_args(parser)
    parser = Task.add_model_specific_args(parser)
    parser = ParsingModel.add_model_specific_args(parser)
    parser = XBertModel.add_model_specific_args(parser)
    parser = Experiment.add_model_specific_args(parser)
    args = parser.parse_args()
    args.task = task
    return args


def probe_all_langs(model="pretrain", src_lang="English", gpu_id=0, verbose=True):
    print(f">>> GPU_ID: {gpu_id}")
    print(f">>> Model: {model}")
    ids = [k for k, _ in ISO638_TO_LANG.items()]
    langs = [v for _, v in ISO638_TO_LANG.items()]
    if model == "pretrain":
        for _, lang in ISO638_TO_LANG.items():
            if lang in ["Catalan", "Czech", "Hungarian", "Urdu", "Tamil", "Chinese"]:
                continue
            res_dict = dict()
            left_layers = 12
            for layer in range(1, 13):
                res_file = os.path.join(REPORT_PATH, f"pretrain_{ids[langs.index(lang)]}_{layer}.json")
                if os.path.exists(res_file):
                    with open(res_file, mode='r', encoding="utf-8") as f:
                        rows = f.readlines()
                        key_list = [list(json.loads(row).keys()) for row in rows]
                        langs_done = [x for row in key_list for x in row]
                        if f"{lang}" in langs_done:
                            print(
                                f">>> {lang}: Layer {layer} has been calculated. \n>>> Skip to next layer...")
                            left_layers -= 1
                            continue
            if left_layers <= 1:
                print(
                    f">>> {lang} has been calculated. \n>>> Skip to next language...")
                continue
            lang_dir = pretrain_analyse_and_path(lang, lang, gpu_id=gpu_id)
            if f"{lang}" not in res_dict:
                res_dict[f"{lang}"] = dict()

            train_path = os.path.join(lang_dir, f"outputs-{ids[langs.index(lang)]}-0.hdf5")
            test_path = os.path.join(lang_dir, f"outputs-{ids[langs.index(lang)]}-2.hdf5")
            for layer in range(1, 13):
                print(f">>> Layer {layer}")
                dataset = HDF5Dataset(filepaths=[train_path, test_path], lang=lang, layer_index=layer)
                for loi in LOIs:
                    accs = probing(dataset, loi=loi, lang=lang, verbose=verbose)
                    res_dict[f"{lang}"][f"{loi}"] = accs
                write_res(res_dict, lang_iso=ids[langs.index(lang)], model="pretrain", layer=layer)
            shutil.rmtree(lang_dir)
    else:
        train_model(src_lang, gpu_id=gpu_id)
        res_dict = dict()
        left_layers = 12
        for layer in range(1, 13):
            res_file = os.path.join(REPORT_PATH, f"finetune_{ids[langs.index(src_lang)]}_{layer}.json")
            if os.path.exists(res_file):
                with open(res_file, mode='r', encoding="utf-8") as f:
                    rows = f.readlines()
                    key_list = [list(json.loads(row).keys()) for row in rows]
                    langs_done = [x for row in key_list for x in row]
                    if f"{src_lang}" in langs_done:
                        print(
                            f">>> {src_lang}: Layer {layer} has been calculated. \n>>> Skip to next layer...")
                        left_layers -= 1
                        continue
        if left_layers == 0:
            print(
                f">>> {src_lang} has been calculated. \n>>> Skip to next language...")
        lang_dir = analyse_and_path(src_lang, src_lang, gpu_id=gpu_id)
        if f"{src_lang}" not in res_dict:
            res_dict[f"{src_lang}"] = dict()
        train_path = os.path.join(lang_dir, f"outputs-{ids[langs.index(src_lang)]}-0.hdf5")
        test_path = os.path.join(lang_dir, f"outputs-{ids[langs.index(src_lang)]}-2.hdf5")
        for layer in range(1, 13):
            print(f">>> Layer {layer}")
            dataset = HDF5Dataset(filepaths=[train_path, test_path], lang=src_lang, layer_index=layer)
            for loi in LOIs:
                accs = probing(dataset, loi=loi, lang=src_lang, verbose=verbose)
                res_dict[f"{src_lang}"][f"{loi}"] = accs
            write_res(res_dict, lang_iso=ids[langs.index(src_lang)], model="fine-tuned", layer=layer)
        shutil.rmtree(lang_dir)

        #for _, tgt_lang in ISO638_TO_LANG.items():
        #    if tgt_lang in ["Catalan", "Czech", "Hungarian", "Urdu", "Tamil", "Chinese"]:
        #        continue
        #    res_dict = dict()
        #    res_file = os.path.join(REPORT_PATH, f"{ids[langs.index(src_lang)]}.json")
        #    lang_pair = [ids[langs.index(lang)] for lang in [src_lang, tgt_lang]]
        #    if os.path.exists(res_file):
        #        with open(res_file, mode='r', encoding="utf-8") as f:
        #            rows = f.readlines()
        #            key_list = [list(json.loads(row).keys()) for row in rows]
        #            lang_pairs_done = [x for row in key_list for x in row]
        #            if f"{lang_pair[0]}2{lang_pair[1]}" in lang_pairs_done:
        #                print(
        #                    f">>> {lang_pair[0]}2{lang_pair[1]} has been calculated. \n>>> Skip to next language...")
        #                continue
        #    tgt_dir = analyse_and_path(src_lang, tgt_lang, gpu_id=gpu_id)
        #    if f"{lang_pair[0]}2{lang_pair[1]}" not in res_dict:
        #        res_dict[f"{lang_pair[0]}2{lang_pair[1]}"] = dict()
        #
        #    train_path = os.path.join(tgt_dir, f"outputs-{ids[langs.index(tgt_lang)]}-0.hdf5")
        #    test_path = os.path.join(tgt_dir, f"outputs-{ids[langs.index(tgt_lang)]}-2.hdf5")
        #    dataset = HDF5Dataset(filepaths=[train_path, test_path], lang=ids[langs.index(tgt_lang)], layer_index=12)
        #    for loi in LOIs:
        #        accs = probing(dataset, loi=loi, lang=ids[langs.index(tgt_lang)], verbose=verbose)
        #        res_dict[f"{lang_pair[0]}2{lang_pair[1]}"][f"{loi}"] = accs
        #    write_res(res_dict, lang_iso=ids[langs.index(src_lang)])
        #    shutil.rmtree(tgt_dir)
    return res_dict


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--model', default="pretrain", type=str)
    argp.add_argument('--src_lang', default="English", type=str)
    argp.add_argument('--gpu_id', default=0, type=int)
    argp.add_argument('--verbose', default=True, type=bool)
    args = argp.parse_args()
    res_dict = probe_all_langs(model=args.model, src_lang=args.src_lang, gpu_id=args.gpu_id, verbose=args.verbose)
    print(res_dict.keys())
