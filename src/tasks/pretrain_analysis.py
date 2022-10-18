from typing import Union, Dict
from argparse import Namespace

from models.pretrained import PreTrainedModel
from dataloader.pretrained import PreTrainedDataLoader
from tasks.base import Task


class PreTrainedAnalysisTask(Task):
    def __init__(self, hparams: Union[Dict, Namespace]):
        super(PreTrainedAnalysisTask, self).__init__(hparams)

    def define_model(self) -> PreTrainedModel:
        return PreTrainedModel(self.hparams)

    def define_dataloader(self) -> PreTrainedDataLoader:
        return PreTrainedDataLoader(
            hparams=self.hparams,
            corpus=self.hparams.corpus,
            tokenizer=self.model.tokenizer
        )

