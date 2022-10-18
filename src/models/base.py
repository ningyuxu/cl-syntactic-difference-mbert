from typing import Union, Dict, List
from argparse import Namespace

from torch import nn

from models.xbert import XBertModel


class Model(nn.Module):
    def __init__(self, hparams: Union[Dict, Namespace]):
        super(Model, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.pretrain_model = self.define_pretrain_model()
        self.pretrain_model.freeze_layers()
        self.tokenizer = self.pretrain_model.tokenizer

    @property
    def data_fields(self) -> List[str]:
        raise NotImplementedError

    def define_pretrain_model(self):
        model = XBertModel(self.hparams)
        return model

    def forward(self, x):
        raise NotImplementedError

