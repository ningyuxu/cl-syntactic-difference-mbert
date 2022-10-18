from argparse import Namespace
from typing import List, Union, Dict, Optional

from transformers import PreTrainedTokenizer

from helper import enumeration as enum
from datasets.base import Dataset
from datasets.label import Label
from datasets.constant import UD_HEAD_LABELS
from dataloader.base import DataLoader
from dataloader.constant import LABEL_PADDING_ID
from dataloader.constant import LANG_PADDING_ID
from dataloader.constant import HEAD_PADDING_ID


class ParsingDataLoader(DataLoader):

    DATA_FIELDS = ["lang", "token_ids", "pos_label_ids", "head_ids", "arc_label_ids"]

    def __init__(
            self,
            hparams: Union[Dict, Namespace],
            corpus: str,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super(ParsingDataLoader, self).__init__(hparams, corpus, tokenizer)
        if hparams.labels:
            self.arc_labels = Label(labels=hparams.labels)
        else:
            self.arc_labels = Label(labels=UD_HEAD_LABELS)

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    @property
    def fields_padding(self) -> Dict:
        return {
            self.data_fields[0]: LANG_PADDING_ID,
            self.data_fields[1]: self.tokenizer.pad_token_id,
            self.data_fields[2]: LABEL_PADDING_ID,
            self.data_fields[3]: HEAD_PADDING_ID,
            self.data_fields[4]: LABEL_PADDING_ID
        }

    @property
    def fields_mapping(self):
        return {
            self.dataset_class.DATA_FIELDS[0]: self.data_fields[0],
            self.dataset_class.DATA_FIELDS[1]: self.data_fields[1],
            self.dataset_class.DATA_FIELDS[2]: self.data_fields[2],
            self.dataset_class.DATA_FIELDS[3]: self.data_fields[3],
            self.dataset_class.DATA_FIELDS[4]: self.data_fields[4]
        }

    def prepare_datasets(self, split: str) -> List[Dataset]:
        if split == enum.Split.train:
            return self.prepare_datasets_helper(
                self.hparams.trn_langs,
                enum.Split.train,
                self.hparams.max_trn_len,
                max_len_unit="subword",
                arc_labels=self.arc_labels
            )
        elif split == enum.Split.dev:
            return self.prepare_datasets_helper(
                self.hparams.val_langs,
                enum.Split.dev,
                self.hparams.max_tst_len,
                max_len_unit="word",
                arc_labels=self.arc_labels
            )
        elif split == enum.Split.test:
            return self.prepare_datasets_helper(
                self.hparams.tst_langs,
                enum.Split.test,
                self.hparams.max_tst_len,
                max_len_unit="word",
                arc_labels=self.arc_labels
            )
        else:
            raise ValueError(f"Unsupported split: {split}")
