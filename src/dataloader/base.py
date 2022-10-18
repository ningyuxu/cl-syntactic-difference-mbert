import re
import os
import json
import hashlib
from argparse import ArgumentParser, Namespace
from functools import partial
from collections import abc
from typing import List, Dict, Union, Optional

import numpy as np
import torch
from torch._six import string_classes
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

from helper import utils
from helper import enumeration as enum
from dataloader.constant import CACHE_PATH
from datasets.base import Dataset
from datasets.ud_treebank import UDTreebankDataset


class DataLoader:
    def __init__(
            self,
            hparams: Union[Dict, Namespace],
            corpus: str,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.corpus = corpus
        self._tokenizer = tokenizer
        self._trn_datasets = None
        self._val_datasets = None
        self._tst_datasets = None
        self._all_datasets = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrain)
        return self._tokenizer

    @property
    def dataset_class(self):
        DatasetClass = {
            enum.Corpus.ud_treebanks: UDTreebankDataset,
        }[self.corpus]
        return DatasetClass

    @property
    def data_fields(self) -> List[str]:
        raise NotImplementedError

    @property
    def fields_padding(self) -> Dict:
        raise NotImplementedError

    @property
    def fields_mapping(self):
        raise NotImplementedError

    def prepare_datasets(self, split: str) -> List[Dataset]:
        raise NotImplementedError

    def prepare_datasets_helper(
            self,
            langs: List[str],
            split: str,
            max_len: int,
            **kwargs
    ) -> List[Dataset]:
        """
        To load data from dataset file according to experiment settings.

        Parameters
        ----------
        langs: languages of data to be loaded
        split: train / valid / test / predict
        max_len: max length of one sentence
        kwargs: extra params for specific subclass

        Returns
        -------
        List of datasets, one dataset for one language
        """
        datasets = []
        for lang in langs:
            # initialize dataset parameters
            params = {
                "model": self.hparams.model,
                "tokenizer": self.tokenizer,
                "lang": lang,
                "split": split,
                "max_len": max_len
            }
            if split == enum.Split.train:
                params["subset_ratio"] = self.hparams.subset_ratio
                params["subset_count"] = self.hparams.subset_count
                params["subset_seed"] = self.hparams.subset_seed
            md5, signature = self._get_signature(params)
            params.update(kwargs)
            del params["model"]
            cache_file = f"{self.hparams.cache_path}/{md5}"
            if self.hparams.cache_dataset and os.path.isfile(cache_file):
                print(f"load from cache {cache_file} with {self.hparams.pretrain}")
                dataset = torch.load(cache_file)
            else:
                dataset = self.dataset_class(**params)
                if self.hparams.cache_dataset:
                    if not os.path.exists(self.hparams.cache_path):
                        os.makedirs(self.hparams.cache_path)
                    print(f"save to cache {cache_file} with {self.hparams.pretrain}")
                    torch.save(dataset, cache_file)
                    with open(f"{cache_file}.json", "w") as fp:
                        json.dump(signature, fp)
            datasets.append(dataset)
        return datasets

    @property
    def trn_datasets(self):
        if self._trn_datasets is None:
            self._trn_datasets = self.prepare_datasets(enum.Split.train)
        return self._trn_datasets

    @property
    def train_data_size(self):
        return sum([len(d) for d in self.trn_datasets])

    @property
    def val_datasets(self):
        if self._val_datasets is None:
            self._val_datasets = self.prepare_datasets(enum.Split.dev)
        return self._val_datasets

    @property
    def val_data_size(self):
        return sum([len(d) for d in self.val_datasets])

    @property
    def tst_datasets(self):
        if self._tst_datasets is None:
            self._tst_datasets = self.prepare_datasets(enum.Split.test)
        return self._tst_datasets

    @property
    def tst_data_size(self):
        return sum([len(d) for d in self.tst_datasets])

    @property
    def all_datasets(self):
        if self._all_datasets is None:
            self._all_datasets = self.prepare_datasets(enum.Split.train)
            self._all_datasets.extend(self.prepare_datasets(enum.Split.dev))
            self._all_datasets.extend(self.prepare_datasets(enum.Split.test))
        return self._all_datasets

    @property
    def all_data_size(self):
        return sum([len(d) for d in self.all_datasets])

    def format(self, datasets: List[Dataset]) -> List[Dataset]:
        """
        Format dataset by setting datasets' field mapping.
        Set self.fields_mapping in subclass to meet specific requirement.
        """
        tgt_datasets = []
        for ds in datasets:
            ds.fields_mapping = self.fields_mapping
            tgt_datasets.append(ds)
        return tgt_datasets

    def load_train_data(self, batch_size: int) -> TorchDataLoader:
        model_datasets = self.format(self.trn_datasets)
        collate = partial(self._padding_collate, padding=self.fields_padding)
        if len(model_datasets) == 1:
            dataset = model_datasets[0]
            sampler = RandomSampler(dataset)
        else:
            dataset = ConcatDataset(model_datasets)
            if self.hparams.mix_sampling:
                sampler = RandomSampler(dataset)
            else:
                sampler = utils.ConcatSampler(dataset, batch_size)
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
        )

    def load_valid_data(self, batch_size: int) -> List[TorchDataLoader]:
        model_datasets = self.format(self.val_datasets)
        collate = partial(self._padding_collate, padding=self.fields_padding)
        return [
            TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate,
            )
            for dataset in model_datasets
        ]

    def load_test_data(self, batch_size: int) -> List[TorchDataLoader]:
        model_datasets = self.format(self.tst_datasets)
        collate = partial(self._padding_collate, padding=self.fields_padding)
        return [
            TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate,
            )
            for dataset in model_datasets
        ]

    def load_all_data(self, batch_size: int) -> List[TorchDataLoader]:
        model_datasets = self.format(self.all_datasets)
        collate = partial(self._padding_collate, padding=self.fields_padding)
        return [
            TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate,
            )
            for dataset in model_datasets
        ]

    @classmethod
    def _get_signature(cls, params: Dict):
        def md5_helper(obj):
            return hashlib.md5(str(obj).encode()).hexdigest()

        signature = dict()
        for key, val in params.items():
            if key == "tokenizer" and isinstance(val, PreTrainedTokenizer):
                signature[key] = md5_helper(list(val.get_vocab().items()))
            else:
                signature[key] = str(val)
        md5 = md5_helper(list(signature.items()))
        return md5, signature

    @classmethod
    def _pad_batch(cls, batch, padding=-1):
        max_len = max([len(b) for b in batch])
        new_batch = []
        for b in batch:
            b_ = np.zeros(max_len, dtype=b.dtype) + padding
            b_[: len(b)] = b
            new_batch.append(b_)
        return new_batch

    @classmethod
    def _padding_collate(cls, batch, padding):
        np_str_obj_array_pattern = re.compile(r"[SaUO]")
        default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, "
            "numpy arrays, numbers, dicts or lists; found {}"
        )
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # if we're in a background process, concatenate directly
                # into a shared memory tensor to avoid an extra copy.
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == "numpy" \
                and elem_type.__name__ != "str_" \
                and elem_type.__name__ != "string_":
            if elem_type.__name__ == "ndarray" \
                    or elem_type.__name__ == "memmap":
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        default_collate_err_msg_format.format(elem.dtype)
                    )
                return cls._padding_collate(
                    [torch.as_tensor(b) for b in cls._pad_batch(batch, padding)],
                    padding
                )  # auto padding
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, abc.Mapping):
            return {
                key: cls._padding_collate([d[key] for d in batch], padding[key])
                for key in elem
            }
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(
                *(cls._padding_collate(samples, padding) for samples in zip(*batch))
            )
        elif isinstance(elem, abc.Sequence):
            transposed = zip(*batch)
            return [cls._padding_collate(samples, padding) for samples in transposed]
        raise TypeError(default_collate_err_msg_format.format(elem_type))

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--max_trn_len", default=128, type=int)
        parser.add_argument("--max_tst_len", default=128, type=int)
        parser.add_argument("--subset_ratio", default=1.0, type=float)
        parser.add_argument("--subset_count", default=-1, type=int)
        parser.add_argument("--subset_seed", default=25, type=int)
        parser.add_argument("--cache_dataset", default=True, type=utils.str2bool)
        parser.add_argument("--cache_path", default=CACHE_PATH, type=str)
        parser.add_argument("--mix_sampling", default=False, type=utils.str2bool)
        return parser
