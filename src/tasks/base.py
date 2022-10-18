import copy
from collections import defaultdict
from argparse import Namespace, ArgumentParser
from typing import Union, Dict, Optional, List

import numpy as np
import torch
from torch import Tensor
import pytorch_lightning as pl

from helper import utils
from helper import enumeration as enum
from helper.metrics import Metric
from models.base import Model
from dataloader.base import DataLoader


class Task(pl.LightningModule):
    """
    Task is designed based on pytorch lightening class for coordinating
    neural network's training validating testing and inference tasks.

    Some basic routing work (data loading, hyper-param optimizing, log
    mechanism are implemented based on pytorch lightening framework.

    Implementing training_step, val_step, test_step, and predict_step for
    specific subclass.
    """
    def __init__(self, hparams: Union[Dict, Namespace]):
        super(Task, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)

        pl.seed_everything(hparams.seed)

        self.model = self.define_model()
        self.dataloader = self.define_dataloader()

        self.optimizer = None
        self.scheduler = None
        self._batch_per_epoch: int = -1
        self._comparison: Optional[str] = None
        self._selection_criterion: Optional[str] = None
        self._metric: Optional[Metric] = None
        self.metrics: Dict[str, Metric] = dict()

    @property
    def batch_per_epoch(self):
        if self._batch_per_epoch < 0:
            total_size = self.dataloader.train_data_size
            self._batch_per_epoch = np.ceil(total_size / self.hparams.batch_size)
        return self._batch_per_epoch

    @property
    def comparison(self):
        assert self._comparison is not None
        return self._comparison

    @property
    def selection_criterion(self):
        assert self._selection_criterion is not None
        return self._selection_criterion

    def setup_metrics(self):
        assert self._metric is not None
        langs = self.hparams.trn_langs + self.hparams.val_langs + self.hparams.tst_langs
        langs = sorted(list(set(langs)))
        for lang in langs:
            self.metrics[lang] = copy.deepcopy(self._metric)
        self.reset_metrics()

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def define_model(self) -> Model:
        raise NotImplementedError

    def define_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        fields = self.model.data_fields
        pretrain_output = self.model.pretrain_model.model_output(batch[fields[1]], batch[fields[0]])

        hidden_states = pretrain_output["hidden_states"]
        hidden_states = utils.tuple_of_tensors_to_tensor(hidden_states)
        hidden_states = torch.permute(hidden_states, (1, 0, 2, 3))

        attentions = pretrain_output["attentions"]
        attentions = utils.tuple_of_tensors_to_tensor(attentions)
        attentions = torch.permute(attentions, (1, 0, 2, 3, 4))

        logger = utils.Logging(self.logger.experiment.log_dir)
        output_list = []
        langs = batch[fields[0]]
        sent = batch[fields[1]]
        pos_label_ids = batch[fields[2]]
        head_ids = batch[fields[3]]
        arc_label_ids = batch[fields[4]]
        for idx in range(len(langs)):
            index = batch_idx * self.hparams.batch_size + idx
            lang = langs[idx]
            token_ids = sent[idx]
            length = torch.count_nonzero(token_ids - self.model.tokenizer.pad_token_id)
            tokens = self.model.tokenizer.convert_ids_to_tokens(token_ids)
            pos_label_ids_c = pos_label_ids[idx]
            head_ids_c = head_ids[idx]
            arc_label_ids_c = arc_label_ids[idx]
            hidden_state = hidden_states[idx]
            attention = attentions[idx]
            output = {
                "dataloader_index": dataloader_idx,
                "index": index,
                "lang": lang,
                "length": length,
                "token_ids": token_ids,
                "tokens": tokens,
                "pos_label_ids": pos_label_ids_c,
                "head_ids": head_ids_c,
                "arc_label_ids": arc_label_ids_c,
                "hidden_state": hidden_state,
                "attention": attention
            }
            logger.save_as_hdf5(output)
            # output_list.append(output)
        return output_list

    def train_dataloader(self):
        return self.dataloader.load_train_data(self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataloader.load_valid_data(self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.dataloader.load_test_data(self.hparams.eval_batch_size)

    def predict_dataloader(self):
        return self.dataloader.load_all_data(self.hparams.eval_batch_size)

    def validation_epoch_end(self, outputs):
        if len(self.hparams.val_langs) == 1:
            outputs = [outputs]
        self.aggregate_outputs(outputs, self.hparams.val_langs, enum.SplitPrefix.valid)
        self.aggregate_metrics(self.hparams.val_langs, enum.SplitPrefix.valid)
        return

    def test_epoch_end(self, outputs):
        if len(self.hparams.tst_langs) == 1:
            outputs = [outputs]
        self.aggregate_outputs(outputs, self.hparams.tst_langs, enum.SplitPrefix.test)
        self.aggregate_metrics(self.hparams.tst_langs, enum.SplitPrefix.test)
        return

    def aggregate_outputs(
            self,
            outputs: List[List[Dict[str, Tensor]]],
            langs: List[str],
            prefix: str
    ):
        assert prefix in [enum.SplitPrefix.valid, enum.SplitPrefix.test]
        aver_result = defaultdict(list)
        for lang, output in zip(langs, outputs):
            for key in output[0]:
                mean_val = torch.stack([x[key] for x in output]).mean()
                self.log(key, mean_val)
                raw_key = key.replace(f"{lang}_", "")
                aver_result[raw_key].append(mean_val)
        for key, vals in aver_result.items():
            self.log(key, torch.stack(vals).mean())

    def aggregate_metrics(self, langs: List[str], prefix: str):
        aver_metric = defaultdict(list)
        non_scalar_metrics = dict()
        for lang in langs:
            metric = self.metrics[lang]
            for key, val in metric.get_metric().items():
                if isinstance(val, torch.Tensor) and val.dim() > 1:
                    non_scalar_metrics[f"{prefix}_{lang}_{key}"] = val
                else:
                    self.log(f"{prefix}_{lang}_{key}", val)
                aver_metric[key].append(val)
        for key, vals in aver_metric.items():
            if isinstance(vals[0], torch.Tensor) and vals[0].dim() > 1:
                non_scalar_metrics[f"{prefix}_{key}"] = torch.mean(torch.stack(vals), 0)
            else:
                self.log(f"{prefix}_{key}", torch.stack(vals).mean())
        logger = utils.Logging(self.logger.experiment.log_dir)
        logger.save_non_scalar_metrics(non_scalar_metrics)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
            "params": [
                p for n, p in self.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.hparams.weight_decay,
        }, {
            "params": [
                p for n, p in self.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )
        warmup_steps, max_steps = self.get_warmup_and_total_steps()
        if self.hparams.schedule == enum.Schedule.invsqroot:
            scheduler = utils.get_inverse_square_root_schedule_with_warmup(
                optimizer, warmup_steps
            )
            interval = "step"
        elif self.hparams.schedule == enum.Schedule.linear:
            scheduler = utils.get_linear_schedule_with_warmup(
                optimizer, warmup_steps, max_steps
            )
            interval = "step"
        elif self.hparams.schedule == enum.Schedule.reduceOnPlateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=0, min_lr=1e-6, mode="min"
            )
            interval = "epoch"
        else:
            raise ValueError(self.hparams.schedule)
        self.optimizer = optimizer
        self.scheduler = scheduler
        scheduler_dict = {"scheduler": scheduler, "interval": interval}
        if self.hparams.schedule == enum.Schedule.reduceOnPlateau:
            scheduler_dict["monitor"] = "val_loss"
        return [optimizer], [scheduler_dict]

    def get_warmup_and_total_steps(self):
        if self.hparams.max_steps is not None:
            max_steps = self.hparams.max_steps
        else:
            max_steps = self.hparams.max_epochs * self.batch_per_epoch
        assert not (
                self.hparams.warmup_steps != -1 and
                self.hparams.warmup_portion != -1
        )
        if self.hparams.warmup_steps != -1:
            assert self.hparams.warmup_steps > 0
            warmup_steps = self.hparams.warmup_steps
        elif self.hparams.warmup_portion != -1:
            assert 0 < self.hparams.warmup_portion < 1
            warmup_steps = int(max_steps * self.hparams.warmup_portion)
        else:
            warmup_steps = 1
        return warmup_steps, max_steps

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--model", choices=enum.Model().choices(), type=str)
        parser.add_argument("--pretrain", choices=enum.PreTrain().choices(), type=str)
        parser.add_argument("--corpus", choices=enum.Corpus().choices(), type=str)
        parser.add_argument("--trn_langs", default=[], nargs="*", type=str)
        parser.add_argument("--val_langs", default=[], nargs="*", type=str)
        parser.add_argument("--tst_langs", default=[], nargs="*", type=str)

        parser.add_argument("--seed", default=25, type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--adam_beta2", default=0.99, type=float)
        parser.add_argument("--adam_eps", default=1e-8, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--eval_batch_size", default=16, type=int)
        parser.add_argument("--schedule", default=enum.Schedule.linear, choices=enum.Schedule().choices(), type=str)
        parser.add_argument("--warmup_steps", default=-1, type=int)
        parser.add_argument("--warmup_portion", default=-1, type=float)
        return parser
