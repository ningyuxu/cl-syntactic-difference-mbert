from typing import Union, Dict
from argparse import Namespace

from helper import enumeration as enum
from helper.metrics import POSTaggingMetric
from models.postagging import POSTaggingModel
from dataloader.postagging import POSTaggingDataLoader
from tasks.base import Task


class POSTaggingTask(Task):
    def __init__(self, hparams: Union[Dict, Namespace]):
        super(POSTaggingTask, self).__init__(hparams)
        self._comparison = "max"
        self._selection_criterion = "val_acc"
        self._metric = POSTaggingMetric(self.model.num_labels)
        self.setup_metrics()

    def define_model(self) -> POSTaggingModel:
        return POSTaggingModel(self.hparams)

    def define_dataloader(self) -> POSTaggingDataLoader:
        return POSTaggingDataLoader(
            hparams=self.hparams,
            corpus=self.hparams.corpus,
            tokenizer=self.model.tokenizer
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self.model.forward(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step_helper(batch, enum.SplitPrefix.valid)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step_helper(batch, enum.SplitPrefix.test)

    def evaluation_step_helper(self, batch, prefix):
        langs = batch[self.model.data_fields[0]]
        loss, log_probs = self.model.forward(batch)
        assert (
                len(set(langs)) == 1
        ), "eval batch should contain only one language"
        lang = batch[self.model.data_fields[0]][0]
        self.metrics[lang].add(batch[self.model.data_fields[2]], log_probs)

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss
        return result
