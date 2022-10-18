from argparse import Namespace
from typing import List, Union, Dict, Optional

from transformers import PreTrainedTokenizer

from helper import enumeration as enum
from datasets.base import Dataset
from datasets.label import Label
from datasets.constant import UD_POS_LABELS
from dataloader.base import DataLoader
from dataloader.constant import LANG_PADDING_ID
from dataloader.constant import LABEL_PADDING_ID


class POSTaggingDataLoader(DataLoader):

    DATA_FIELDS = ["lang", "token_ids", "pos_label_ids"]

    def __init__(
            self,
            hparams: Union[Dict, Namespace],
            corpus: str,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super(POSTaggingDataLoader, self).__init__(hparams, corpus, tokenizer)
        if hparams.labels:
            self.pos_labels = Label(labels=hparams.labels)
        else:
            self.pos_labels = Label(labels=UD_POS_LABELS)

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    @property
    def fields_padding(self) -> Dict:
        return {
            self.data_fields[0]: LANG_PADDING_ID,
            self.data_fields[1]: self.tokenizer.pad_token_id,
            self.data_fields[2]: LABEL_PADDING_ID,
        }

    @property
    def fields_mapping(self):
        return {
            self.dataset_class.DATA_FIELDS[0]: self.data_fields[0],
            self.dataset_class.DATA_FIELDS[1]: self.data_fields[1],
            self.dataset_class.DATA_FIELDS[2]: self.data_fields[2]
        }

    def prepare_datasets(self, split: str) -> List[Dataset]:
        if split == enum.Split.train:
            return self.prepare_datasets_helper(
                self.hparams.trn_langs,
                enum.Split.train,
                self.hparams.max_trn_len,
                max_len_unit="subword",
                pos_labels=self.pos_labels,
            )
        elif split == enum.Split.dev:
            return self.prepare_datasets_helper(
                self.hparams.val_langs,
                enum.Split.dev,
                self.hparams.max_tst_len,
                max_len_unit="word",
                pos_labels=self.pos_labels,
            )
        elif split == enum.Split.test:
            return self.prepare_datasets_helper(
                self.hparams.tst_langs,
                enum.Split.test,
                self.hparams.max_tst_len,
                max_len_unit="word",
                pos_labels=self.pos_labels,
            )
        else:
            raise ValueError(f"Unsupported split: {split}")
