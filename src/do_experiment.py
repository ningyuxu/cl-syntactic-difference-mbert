import argparse
from typing import List
from argparse import Namespace

import torch
from transformers import logging

from helper import enumeration as enum
from datasets import label
from datasets.base import ISO638_TO_LANG
from datasets.constant import UD_POS_LABELS
from datasets.constant import UD_HEAD_LABELS
from dataloader.base import DataLoader
from models.parsing import ParsingModel
from tasks.base import Task
from models.xbert import XBertModel
from experiment.experiment import Experiment

logging.set_verbosity_error()


def get_choices(
        description: str,
        values: List[str],
        multiple_choice: bool = True,
        give_all_option: bool = True,
        with_other_label: bool = False
) -> List[str]:
    """
    Methods designed to get user's choices.

    Parameters
    ----------
    description: prompting message
    values: choices list
    multiple_choice: can select multiple choices?
    give_all_option: can select all?
    with_other_label: with ``other`` option?

    Returns
    -------
    user's choices
    """
    choices: List[str] = []

    ids_items = [f"{i}: {v}" for i, v in enumerate(values)]
    if give_all_option:
        ids_items.append(f"{len(ids_items)}: ALL")

    values_ids = []
    while True:
        prompt = f"\n{description} \n{ids_items}: \n"
        try:
            values_ids = [int(idx) for idx in input(prompt).split()]
            if len(values_ids) == 0:
                raise ValueError
            if not multiple_choice and len(values_ids) > 1:
                raise ValueError
            for idx in values_ids:
                if idx < 0 or idx > len(ids_items):
                    raise ValueError
            break
        except ValueError:
            print("Invalid input.")
            continue

    if give_all_option and len(values) in values_ids:
        choices = values
    else:
        for idx in values_ids:
            choices.append(values[idx])
        if with_other_label:
            choices.append(label.LABEL_OTHERS)

    return choices


def get_model(task: str) -> str:
    # analyse is the only task we can perform on pretrain model
    models = enum.Model().choices() if task == enum.Task.analyse else enum.Model().choices()[1:]
    if len(models) == 1:
        return models[0]
    choices = get_choices(
        description="Select Model",
        values=models,
        multiple_choice=False,
        give_all_option=False,
        with_other_label=False
    )
    return choices[0]


def get_pretrain() -> str:
    pretrains = enum.PreTrain().choices()
    if len(pretrains) == 1:
        return pretrains[0]
    choices = get_choices(
        description="Select PreTrained Model",
        values=pretrains,
        multiple_choice=False,
        give_all_option=False,
        with_other_label=False
    )
    return choices[0]


def get_corpus() -> str:
    corpus = enum.Corpus().choices()
    if len(corpus) == 1:
        return corpus[0]
    choices = get_choices(
        description="Select Corpus",
        values=corpus,
        multiple_choice=False,
        give_all_option=False,
        with_other_label=False
    )
    return choices[0]


def get_langs(split: str) -> List[str]:
    ids = [k for k, _ in ISO638_TO_LANG.items()]
    langs = [v for _, v in ISO638_TO_LANG.items()]
    if split == enum.Split.train:
        desc = "Select Source Languages"
    elif split == enum.Split.test:
        desc = "Select Target Languages"
    else:
        raise ValueError(f"Unsupported split {split}")
    choices = get_choices(
        description=desc,
        values=langs,
        multiple_choice=True,
        give_all_option=True,
        with_other_label=False
    )
    return [ids[langs.index(lang)] for lang in choices]


def get_labels(model: str) -> List[str]:
    if model == enum.Model.postagging:
        pos_labels = UD_POS_LABELS
        return get_choices(
            description="Select POS-Tagging Labels",
            values=pos_labels,
            multiple_choice=True,
            give_all_option=True,
            with_other_label=True
        )
    if model == enum.Model.parsing:
        head_labels = UD_HEAD_LABELS
        return get_choices(
            description="Select Dependency Labels",
            values=head_labels,
            multiple_choice=True,
            give_all_option=True,
            with_other_label=True
        )


def get_gpu():
    num_device = torch.cuda.device_count()
    while True:
        try:
            gpu_id = int(input("gpu id: "))
            if gpu_id < 0 or gpu_id >= num_device:
                raise ValueError("Invalid gpu id.")
            break
        except ValueError:
            print("Invalid gpu id.")
    return gpu_id


def user_interaction(hparams: Namespace) -> Experiment:
    exp = Experiment(hparams)
    if hparams.task == enum.Task.train:
        exp.model = get_model(hparams.task)
        exp.pretrain = get_pretrain()
        exp.corpus = get_corpus()
        exp.src_langs = get_langs(enum.Split.train)
        exp.tgt_langs = exp.src_langs
        exp.data_labels = get_labels(exp.model)
    elif hparams.task == enum.Task.test:
        exp.model = get_model(hparams.task)
        exp.pretrain = get_pretrain()
        exp.corpus = get_corpus()
        exp.src_langs = get_langs(enum.Split.train)
        exp.tgt_langs = get_langs(enum.Split.test)
        exp.data_labels = get_labels(exp.model)
    elif hparams.task == enum.Task.analyse:
        exp.model = get_model(hparams.task)
        exp.pretrain = get_pretrain()
        exp.corpus = get_corpus()
        if exp.model == enum.Model.pretrain:
            exp.tgt_langs = get_langs(enum.Split.test)
        else:
            exp.src_langs = get_langs(enum.Split.train)
            exp.data_labels = get_labels(exp.model)
            exp.tgt_langs = get_langs(enum.Split.test)
    return exp


def setup_experiment(hparams: Namespace) -> Experiment:
    exp = user_interaction(hparams)
    confirmed = False
    while not confirmed:
        src_langs = [ISO638_TO_LANG[lang] for lang in exp.src_langs]
        tgt_langs = [ISO638_TO_LANG[lang] for lang in exp.tgt_langs]
        desc = "Accept experiment setting? \n"
        desc += "-------------------------------------------------------- \n"
        desc += f"* Experiment Model: {exp.model} \n"
        desc += f"* PreTrained Model: {exp.pretrain} \n"
        desc += f"* Corpus: {exp.corpus} \n"
        if hparams.task == enum.Task.train:
            desc += f"* Source Languages: {src_langs} \n"
            desc += f"* Data Labels: {exp.data_labels} \n"
        if hparams.task == enum.Task.test:
            desc += f"* Source Languages: {src_langs} \n"
            desc += f"* Target Languages: {tgt_langs} \n"
            desc += f"* Data Labels: {exp.data_labels} \n"
        if hparams.task == enum.Task.analyse:
            if exp.model == enum.Model.pretrain:
                desc += f"* Target Languages: {tgt_langs} \n"
            else:
                desc += f"* Source Languages: {src_langs} \n"
                desc += f"* Data Labels: {exp.data_labels} \n"
                desc += f"* Target Languages: {tgt_langs} \n"
        desc += "--------------------------------------------------------"
        yes_or_no = ["Yes", "No"]
        choice = get_choices(
            description=desc,
            values=yes_or_no,
            multiple_choice=False,
            give_all_option=False,
            with_other_label=False
        )
        if choice[0] == "No":
            confirmed = False
            exp = user_interaction(hparams.task)
        else:
            confirmed = True
            exp.init_model_home()
    return exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=enum.Task().choices())
    parser = DataLoader.add_model_specific_args(parser)
    parser = Task.add_model_specific_args(parser)
    parser = ParsingModel.add_model_specific_args(parser)
    parser = XBertModel.add_model_specific_args(parser)
    parser = Experiment.add_model_specific_args(parser)

    args = parser.parse_args()
    experiment = setup_experiment(args)
    gpus = get_gpu()
    experiment.run([gpus])
