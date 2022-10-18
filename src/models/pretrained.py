from typing import Union, Dict, List
from argparse import Namespace

from dataloader.pretrained import PreTrainedDataLoader
from models.base import Model


class PreTrainedModel(Model):
    def __init__(self, hparams: Union[Dict, Namespace]):
        super(PreTrainedModel, self).__init__(hparams)

    @property
    def data_fields(self) -> List[str]:
        return PreTrainedDataLoader.DATA_FIELDS

    def forward(self, x):
        lang = x[self.data_fields[0]]
        token_ids = x[self.data_fields[1]]
        emb = self.pretrain_model.encode_sent(sent=token_ids, langs=lang)
        return emb
