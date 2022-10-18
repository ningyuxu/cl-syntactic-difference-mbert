import json
import os
import glob
import hashlib
from typing import List, Optional
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from helper import utils
from helper import enumeration as enum
from experiment.constant import EXPS_PATH
from tasks.base import Task
from tasks.pretrain_analysis import PreTrainedAnalysisTask
from tasks.postagging_all import POSTaggingTask
from tasks.parsing_all import ParsingTask


class Experiment:
    """
    Class designed for cross lingual transfer experiment. Responsible for parsing
    experiment's parameters from user input; initializing components such as
    trainer, model, logger ... to start the experiment; preparing environment such
    as make and empty experiment directory.

    Key properties:
        task: what to do? [train / test / analyse]
        model: fine-tune model for specific downstream task, such as
            [pretrain / postagging / parsing / ...], "pretrain“ means no downstream
            task, just do something on pretrained model.
        pretrain: pretrained model, such as [bert-base-multilingual-cased ...]
        corpus: which corpus(datasets) will be used?
        src_langs: list of model's training languages
        tgt_langs: list of languages for testing / analysing
        data_labels: labels of data such as pos-tagging label for downstream task
    """
    def __init__(self, hparams: Namespace):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.model: Optional[str] = None
        self.pretrain: Optional[str] = None
        self.corpus: Optional[str] = None
        self.src_langs: Optional[List[str]] = []
        self.tgt_langs: Optional[List[str]] = []
        self.data_labels: Optional[List[str]] = []

        self.root_path: str = EXPS_PATH
        self._model_dir: Optional[str] = None
        self._version_dir: Optional[str] = None
        self._log_path: Optional[str] = None
        self._logging_callback: Optional[Callback] = None
        self._early_stopping_callback: Optional[Callback] = None
        self._checkpoint_callback: Optional[Callback] = None
        self._task: Optional[Task] = None
        self._logger: Optional[TensorBoardLogger] = None
        self._trainer: Optional[Trainer] = None
        self._callbacks: Optional[List[Callback]] = []

    @property
    def model_dir(self) -> str:
        if not self._model_dir:
            signature = dict()
            signature["model"] = self.md5_helper(self.model)
            signature["pretrain"] = self.md5_helper(self.pretrain)
            signature["corpus"] = self.md5_helper(self.corpus)
            signature["src_langs"] = self.md5_helper(self.src_langs)
            signature["data_labels"] = self.md5_helper(self.data_labels)
            self._model_dir = self.model + '-' + self.md5_helper(signature)
        return self._model_dir

    @property
    def version_dir(self) -> str:
        if not self._version_dir:
            if self.hparams.task == enum.Task.train:
                self._version_dir = f"version_{self.hparams.task}"
            elif self.hparams.task in [enum.Task.test, enum.Task.analyse]:
                signature = dict()
                signature["corpus"] = self.md5_helper(self.corpus)
                signature["tgt_langs"] = self.md5_helper(self.tgt_langs)
                signature["data_labels"] = self.md5_helper(self.data_labels)
                self._version_dir = f"version_{self.hparams.task}_" + self.md5_helper(signature)
            else:
                raise ValueError(f"Unsupported task {self.hparams.task}.")
        return self._version_dir

    @property
    def log_path(self) -> str:
        if not self._log_path:
            self._log_path = os.path.join(self.root_path, self.model_dir, self.version_dir)
        return self._log_path

    @property
    def logger(self) -> TensorBoardLogger:
        if not self._logger:
            self._logger = TensorBoardLogger(self.root_path, name=self.model_dir, version=self.version_dir)
        return self._logger

    @property
    def task(self) -> Task:
        if not self._task:
            TaskClass = {
                enum.Model.pretrain: PreTrainedAnalysisTask,
                enum.Model.postagging: POSTaggingTask,
                enum.Model.parsing: ParsingTask,
            }[self.model]

            self.hparams.model = self.model
            self.hparams.pretrain = self.pretrain
            self.hparams.corpus = self.corpus
            self.hparams.labels = self.data_labels

            if self.hparams.task == enum.Task.train:
                self.hparams.trn_langs = self.src_langs
                self.hparams.val_langs = self.src_langs
                task = TaskClass(self.hparams)
            elif self.hparams.task == enum.Task.test:
                checkpoint = self.find_checkpoint()
                assert os.path.isfile(checkpoint), "No checkpoint found for the model."
                task = TaskClass.load_from_checkpoint(
                    checkpoint, tst_langs=self.tgt_langs
                )
            elif self.hparams.task == enum.Task.analyse:
                if self.hparams.model == enum.Model.pretrain:
                    self.hparams.trn_langs = self.tgt_langs
                    self.hparams.val_langs = self.tgt_langs
                    self.hparams.tst_langs = self.tgt_langs
                    task = TaskClass(self.hparams)
                else:
                    checkpoint = self.find_checkpoint()
                    assert os.path.isfile(checkpoint), "No checkpoint found for the model."
                    task = TaskClass.load_from_checkpoint(
                        checkpoint,
                        trn_langs=self.tgt_langs,
                        val_langs=self.tgt_langs,
                        tst_langs=self.tgt_langs
                    )
            else:
                raise ValueError(f"Unsupported action {self.hparams.task}")
            self._task = task
        return self._task

    @property
    def logging_callback(self) -> Callback:
        if not self._logging_callback:
            log_path = os.path.join(self.root_path, self.model_dir, self.version_dir)
            self._logging_callback = utils.Logging(log_path)
        return self._logging_callback

    @property
    def early_stopping_callback(self) -> Callback:
        if not self._early_stopping_callback:
            self._early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor=self.task.selection_criterion,
                min_delta=self.hparams.min_delta,
                patience=self.hparams.patience,
                verbose=True,
                mode=self.task.comparison,
                strict=True
            )
        return self._early_stopping_callback

    @property
    def checkpoint_callback(self) -> Callback:
        if not self._checkpoint_callback:
            ckpts_path = os.path.join(self.root_path, self.model_dir, self.version_dir, "ckpts")
            ckpts_file = "ckpts_{epoch}-{%s:.3f}" % self.task.selection_criterion
            self._checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpts_path,
                filename=ckpts_file,
                monitor=self.task.selection_criterion,
                verbose=True,
                save_last=self.hparams.save_last,
                save_top_k=self.hparams.save_top_k,
                mode=self.task.comparison,
            )
        return self._checkpoint_callback

    @property
    def callbacks(self):
        if not self._callbacks:
            self._callbacks = [self.logging_callback]
            if self.hparams.task == enum.Task.train:
                self._callbacks.append(self.early_stopping_callback)
                self._callbacks.append(self.checkpoint_callback)
        return self._callbacks

    @classmethod
    def md5_helper(cls, obj):
        return hashlib.md5(str(obj).encode()).hexdigest()

    def init_model_home(self):
        model_path = os.path.join(self.root_path, self.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            desc_file = os.path.join(model_path, "README")
            with open(desc_file, mode='w', encoding="utf-8") as f:
                description = {
                    "model": self.model,
                    "pretrain": self.pretrain,
                    "corpus": self.corpus,
                    "src_langs": self.src_langs,
                    "labels": self.data_labels
                }
                json.dump(description, f)

    def find_checkpoint(self):
        checkpoint = ""
        train_path = os.path.join(self.root_path, self.model_dir, f"version_{enum.Task.train}")
        if os.path.exists(train_path):
            path = os.path.join(train_path, "ckpts")
            fp = f"{path}/ckpts_*.ckpt"
            _fp = glob.glob(fp)
            if len(_fp) == 1:
                checkpoint = _fp[0]
        return checkpoint

    def run(self, gpus: List[int]):
        trainer = pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            default_root_dir=self.root_path,
            gradient_clip_val=self.hparams.gradient_clip_val,
            num_nodes=self.hparams.num_nodes,
            gpus=gpus,
            auto_select_gpus=True,
            overfit_batches=self.hparams.overfit_batches,
            track_grad_norm=self.hparams.track_grad_norm,
            check_val_every_n_epoch=self.hparams.check_val_every_n_epoch,
            fast_dev_run=self.hparams.fast_dev_run,
            accumulate_grad_batches=self.hparams.accumulate_grad_batches,
            max_epochs=self.hparams.max_epochs,
            min_epochs=self.hparams.min_epochs,
            max_steps=self.hparams.max_steps,
            min_steps=self.hparams.min_steps,
            val_check_interval=int(self.hparams.val_check_interval)
            if self.hparams.val_check_interval > 1 else self.hparams.val_check_interval,
            log_every_n_steps=self.hparams.log_every_n_steps,
            accelerator=self.hparams.accelerator,
            precision=self.hparams.precision,
            resume_from_checkpoint=self.hparams.resume_from_checkpoint,
            replace_sampler_ddp=True,
            terminate_on_nan=True,
            amp_backend="apex",
            amp_level="O1",
        )
        if self.hparams.task == enum.Task.train:
            trainer.fit(model=self.task)
        if self.hparams.task == enum.Task.test:
            trainer.test(model=self.task)
        if self.hparams.task == enum.Task.analyse:
            trainer.predict(model=self.task)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--min_delta", default=1e-3, type=float)
        parser.add_argument("--patience", default=5, type=int)
        parser.add_argument("--save_last", default=False, type=utils.str2bool)
        parser.add_argument("--save_top_k", default=1, type=int)
        parser.add_argument("--gradient_clip_val", default=0, type=float)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--overfit_batches", default=0.0, type=float)
        parser.add_argument("--track_grad_norm", default=-1, type=int)
        parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
        parser.add_argument("--fast_dev_run", default=False, type=utils.str2bool)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--max_epochs", default=10, type=int)
        parser.add_argument("--min_epochs", default=1, type=int)
        parser.add_argument("--max_steps", default=None, type=int)
        parser.add_argument("--min_steps", default=None, type=int)
        parser.add_argument("--val_check_interval", default=1.0, type=float)
        parser.add_argument("--log_every_n_steps", default=10, type=int)
        parser.add_argument("--accelerator", default=None, type=str)
        parser.add_argument("--precision", default=32, type=int)
        parser.add_argument("--resume_from_checkpoint", default=None, type=str)
        parser.add_argument("--labels", default=[], nargs="*", type=str)
        return parser
