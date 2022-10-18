import json
import os
import shutil
import argparse
from argparse import Namespace

import numpy as np
from transformers import logging

from measure_discrepancy import a_distance, otce, otdd
from measure_discrepancy.utils import write_dist, REPORT_PATH

from datasets.constant import UD_HEAD_LABELS, UD_ISO638_TO_LANG, ISO638_TO_LANG
from dataloader.base import DataLoader
from models.parsing import ParsingModel
from tasks.base import Task
from models.xbert import XBertModel
from experiment.experiment import Experiment
from experiment import experiment
import helper.enumeration as enum
from measure_discrepancy.a_distance import HDF5Dataset


logging.set_verbosity_error()


def get_args(task):
    parser = argparse.ArgumentParser()
    # parser.add_argument("task", type=str, default=task)
    parser.add_argument('--dist_kind', default="otdd", type=str)
    parser.add_argument('--src_lang', default="Hebrew", type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser = DataLoader.add_model_specific_args(parser)
    parser = Task.add_model_specific_args(parser)
    parser = ParsingModel.add_model_specific_args(parser)
    parser = XBertModel.add_model_specific_args(parser)
    parser = Experiment.add_model_specific_args(parser)
    args = parser.parse_args()
    args.task = task
    return args


def analyse_and_path(src_lang, tgt_lang, gpu_id=0):
    task = enum.Task().analyse
    hparams = get_args(task)
    exp = Experiment(hparams)
    exp.model = enum.Model().pretrain
    exp.pretrain = enum.PreTrain.mbert
    exp.corpus = enum.Corpus.ud_treebanks
    ids = [k for k, _ in ISO638_TO_LANG.items()]
    langs = [v for _, v in ISO638_TO_LANG.items()]
    if exp.model == enum.Model().pretrain:
        exp.tgt_langs = [ids[langs.index(lang)] for lang in [tgt_lang]]
    else:
        exp.src_langs = [ids[langs.index(lang)] for lang in [src_lang]]
        exp.tgt_langs = [ids[langs.index(lang)] for lang in [tgt_lang]]
        exp.data_labels = UD_HEAD_LABELS
    signature = dict()
    signature["corpus"] = exp.md5_helper(exp.corpus)
    signature["tgt_langs"] = exp.md5_helper(exp.tgt_langs)
    signature["data_labels"] = exp.md5_helper(exp.data_labels)
    file_path = os.path.join(exp.root_path, exp.model_dir,
                             f"version_{enum.Task.analyse}_{exp.md5_helper(signature)}")
    if os.path.exists(file_path):
        return file_path
    else:
        exp.init_model_home()
        exp.run([gpu_id])
        return file_path


def get_all_dists(dist_kind, src_lang, gpu_id=5, verbose=True, layer=0):
    print(f">>> GPU_ID: {gpu_id}")
    print(f">>> dist_kind: {dist_kind}")
    src_dir = analyse_and_path(src_lang, src_lang, gpu_id=gpu_id)
    ids = [k for k, _ in ISO638_TO_LANG.items()]
    langs = [v for _, v in ISO638_TO_LANG.items()]
    dist_dict = dict()
    for _, tgt_lang in ISO638_TO_LANG.items():
        if tgt_lang == src_lang or tgt_lang in ["Catalan", "Czech", "Hungarian", "Urdu", "Tamil",
                                                "Chinese"]:
            continue
        dist_dict = dict()
        res_file = os.path.join(REPORT_PATH, f"{dist_kind}_pretrain_{ids[langs.index(src_lang)]}_{layer}.json")
        lang_pair = [ids[langs.index(lang)] for lang in [src_lang, tgt_lang]]
        if os.path.exists(res_file):
            with open(res_file, mode='r', encoding="utf-8") as f:
                rows = f.readlines()
                key_list = [list(json.loads(row).keys()) for row in rows]
                lang_pairs_done = [x for row in key_list for x in row]
                if f"{lang_pair[0]}_to_{lang_pair[1]}" in lang_pairs_done or \
                        f"{lang_pair[1]}_to_{lang_pair[0]}" in lang_pairs_done:
                    print(f">>> {lang_pair[0]}_to_{lang_pair[1]} has been calculated. \n>>> Skip to next language...")
                    continue
        tgt_dir = analyse_and_path(src_lang, tgt_lang, gpu_id=gpu_id)
        if f"{lang_pair[0]}_to_{lang_pair[1]}" not in dist_dict:
            dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"] = dict()

        if dist_kind == "a-distance":
            src_path_un = os.path.join(src_dir, f"outputs-{ids[langs.index(src_lang)]}-1.hdf5")
            tgt_path_un = os.path.join(tgt_dir, f"outputs-{ids[langs.index(tgt_lang)]}-1.hdf5")
            dataset_un = HDF5Dataset(filepaths=[src_path_un, tgt_path_un], langs=lang_pair, layer_index=layer)
            src_path = os.path.join(src_dir, f"outputs-{ids[langs.index(src_lang)]}-0.hdf5")
            tgt_path = os.path.join(tgt_dir, f"outputs-{ids[langs.index(tgt_lang)]}-0.hdf5")
            dataset = HDF5Dataset(filepaths=[src_path, tgt_path], langs=lang_pair, layer_index=layer)
            for loi in a_distance.LOIs:
                if loi == "uas" or loi == "un":
                    dist = a_distance.compute_a_dist_with_dataset(dataset_un, loi=loi, langs=lang_pair)
                else:
                    dist = a_distance.compute_a_dist_with_dataset(dataset, loi=loi, langs=lang_pair)
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"{loi}"] = dist
        elif dist_kind == "otce":
            for loi in otce.LOIs:
                src_path = os.path.join(src_dir, f"outputs-{ids[langs.index(src_lang)]}-0.hdf5")
                tgt_path = os.path.join(tgt_dir, f"outputs-{ids[langs.index(tgt_lang)]}-0.hdf5")
                if loi == "uas":
                    w, ce = otce.compute_w_ce(filepaths=[src_path, tgt_path], langs=lang_pair,
                                              loi=loi, layer_index=layer, kind="uas",
                                              device=f"cuda:{gpu_id}", verbose=verbose)
                else:
                    w, ce = otce.compute_w_ce(filepaths=[src_path, tgt_path], langs=lang_pair,
                                              loi=loi, layer_index=layer, kind="las",
                                              device=f"cuda:{gpu_id}", verbose=verbose)
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"w_{loi}"] = w
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"ce_{loi}"] = ce
        elif dist_kind == "otdd":
            for loi in otdd.LOIs:
                src_path = os.path.join(src_dir, f"outputs-{ids[langs.index(src_lang)]}-0.hdf5")
                tgt_path = os.path.join(tgt_dir, f"outputs-{ids[langs.index(tgt_lang)]}-0.hdf5")
                d, label_dist, src_classes, tgt_classes = otdd.compute_otdd(
                    filepaths=[src_path, tgt_path], langs=lang_pair, loi=loi,
                    layer_index=layer, maxsize=2000, maxsamples=1500, device=f"cuda:{gpu_id}"
                )
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"{loi}"] = d
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"{loi}_label_dist"] = label_dist
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"{loi}_src_classes"] = src_classes
                dist_dict[f"{lang_pair[0]}_to_{lang_pair[1]}"][f"{loi}_tgt_classes"] = tgt_classes
        write_dist(dist_dict, src_lang_iso=ids[langs.index(src_lang)],
                   dist_kind=dist_kind, model="pretrain", layer=layer)
        shutil.rmtree(tgt_dir)
    shutil.rmtree(src_dir)
    return dist_dict


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--dist_kind', default="otdd", type=str)
    argp.add_argument('--src_lang', default="Hebrew", type=str)
    argp.add_argument('--gpu_id', default=0, type=int)
    argp.add_argument('--verbose', default=True, type=bool)
    args = argp.parse_args()
    dist_dict = get_all_dists(args.dist_kind, args.src_lang, gpu_id=args.gpu_id, verbose=args.verbose)
    print(dist_dict.keys())
