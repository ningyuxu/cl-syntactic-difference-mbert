import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict
import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import LambdaLR

from pytorch_lightning import Callback

from transformers import AutoTokenizer, AutoModel, AutoConfig


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Logging(Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.filename = os.path.join(save_dir, "results.json")

    def on_validation_start(self, trainer, pl_module):
        pl_module.reset_metrics()

    def on_validation_end(self, trainer, pl_module):
        with open(self.filename, mode='a', encoding="utf-8") as f:
            logs = dict()
            for k, v in trainer.callback_metrics.items():
                if k.startswith("val_"):
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 0:
                            v = v.item()
                        else:
                            v = v.tolist()
                    logs[k] = v
            logs["step"] = trainer.global_step
            print(json.dumps(logs), file=f)

    def on_test_start(self, trainer, pl_module):
        pl_module.reset_metrics()

    def on_test_end(self, trainer, pl_module):
        with open(self.filename, mode='a', encoding="utf-8") as f:
            logs = dict()
            for k, v in trainer.callback_metrics.items():
                if k.startswith("tst_") or k == "select":
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    logs[k] = v
            # assert "select" in logs
            print(json.dumps(logs), file=f)

    def save_non_scalar_metrics(self, metrics):
        with open(self.filename, mode='a', encoding="utf-8") as f:
            logs = dict()
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        v = v.item()
                    else:
                        v = v.tolist()
                logs[k] = v
            print(json.dumps(logs), file=f)

    def save_as_hdf5(self, metrics):
        hdf5filename = os.path.join(self.save_dir,
                                    f"outputs-{metrics['lang']}-{metrics['dataloader_index']}.hdf5")
        with h5py.File(hdf5filename, 'a') as f:
            grp = f.create_group(str(metrics["index"]))
            for k, v in metrics.items():
                if k == "index" or k == "dataloader_index" or k == "lang":
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                    dset = grp.create_dataset(k, data=v)
                elif isinstance(v, str):
                    dt = h5py.string_dtype(encoding='utf-8')
                    dset = grp.create_dataset(k, data=v, dtype=dt)
                else:
                    assert isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, int), \
                        "unsupported data type"
                    if isinstance(v[0], str):
                        v = [w.encode() for w in v]
                        dt = h5py.special_dtype(vlen=str)
                        dset = grp.create_dataset(k, shape=(len(v),), dtype=dt, data=v)
                    else:
                        v = np.array(v)
                        dset = grp.create_dataset(k, data=v)


class MappingCheckpoint(Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.filename = os.path.join(save_dir, "mapping.pth")
        self.mappings = nn.ModuleList([])
        self.mappings_best: Dict[str, float] = dict()

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        if (
            pl_module.hparams.task == "alignment"
            and pl_module.hparams.aligner_sim == "linear"
        ):
            for _ in range(pl_module.num_layers):
                m = nn.Linear(pl_module.hidden_size, pl_module.hidden_size, bias=False)
                self.mappings.append(m)

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        if (
            pl_module.hparams.task == "alignment"
            and pl_module.hparams.aligner_sim == "linear"
        ):
            metrics = trainer.callback_metrics
            new_best_mappings = []
            for i, mapping in enumerate(pl_module.mappings):
                key = f"val_layer{i}_loss"
                if key not in self.mappings_best or (
                    self.mappings_best[key] > metrics[key]
                ):
                    new_best_mappings.append(i)
                    self.mappings_best[key] = metrics[key]
                    self.mappings[i].load_state_dict(mapping.state_dict())

            if new_best_mappings:
                print(
                    f"found new best mappings at {new_best_mappings} in step {trainer.global_step}"
                )
                torch.save(self.mappings, self.filename)


def get_inverse_square_root_schedule_with_warmup(
    optimizer, warmup_steps, last_epoch=-1
):
    """
    Create a schedule with linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def lr_lambda(step):
        decay_factor = warmup_steps ** 0.5
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return decay_factor * step ** -0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, warmup_steps, training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        ratio = (training_steps - step) / max(1, training_steps - warmup_steps)
        return max(ratio, 0)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def pad_batch(batch, padding=-1):
    max_len = max([len(b) for b in batch])
    new_batch = []
    for b in batch:
        b_ = np.zeros(max_len, dtype=b.dtype) + padding
        b_[: len(b)] = b
        new_batch.append(b_)
    return new_batch


np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


class ConcatSampler(Sampler):
    def __init__(self, concat_dataset: ConcatDataset, samples_per_dataset: int):
        assert isinstance(concat_dataset, ConcatDataset)
        super(ConcatSampler, self).__init__(concat_dataset)
        self.concat_dataset = concat_dataset
        self.nb_datasets = len(concat_dataset.datasets)
        self.samples_per_dataset = samples_per_dataset

        weight = torch.tensor([len(ds) for ds in concat_dataset.datasets]).float()
        self.weight = weight / weight.sum()

    def sample_dataset(self):
        return torch.multinomial(self.weight, 1, replacement=True).item()

    def __iter__(self):
        iterators = [iter(RandomSampler(ds)) for ds in self.concat_dataset.datasets]
        done = np.array([False] * self.nb_datasets)
        while not done.all():
            dataset_id = self.sample_dataset()
            if done[dataset_id]:
                continue
            batch = []
            for _ in range(self.samples_per_dataset):
                try:
                    idx = next(iterators[dataset_id])
                except StopIteration:
                    done[dataset_id] = True
                    break
                if dataset_id > 0:
                    idx += self.concat_dataset.cumulative_sizes[dataset_id - 1]
                batch.append(idx)

            if len(batch) == self.samples_per_dataset:
                yield from batch

    def __len__(self):
        n = self.samples_per_dataset
        return sum([len(ds) // n * n for ds in self.concat_dataset.datasets])


def masked_log_softmax(
    vector: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of
    ``vector`` should be masked.  This performs a log_softmax on just the non-masked
    portions of ``vector``.  Passing ``None`` in for the mask is also acceptable; you'll
    just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that
    ``mask`` is broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions
    than ``vector``, we will unsqueeze on dimension 1 until they match.  If you need a
    different unsqueezing of your mask, do it yourself before passing the mask into this
    function.
    In the case that the input vector is completely masked, the return value of this
    function is arbitrary, but not ``nan``.  You should be masking the result of whatever
    computation comes out of this in that case, anyway, so the specific values returned
    shouldn't matter.  Also, the way that we deal with this case relies on having
    single-precision floats; mixing half-precision floats with fully-masked vectors will
    likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector
    is -50 or lower), the way we handle masking here could mess you up.  But if you've
    got logit values that extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but
        # it results in nans when the whole vector is masked.  We need a very small value
        # instead of a zero in the mask for these cases.  log(1 + 1e-45) is still
        # basically 0, so we can safely just add 1e-45 before calling mask.log().  We use
        # 1e-45 because 1e-46 is so small it becomes 0 - this is just the smallest value
        # we can actually use.
        vector = vector + (mask + 1e-45).log()
    return F.log_softmax(vector, dim=dim)


def get_hgfs_model(
        pretrain_name: str, model_dir: Path, output_hidden_states: bool = True, output_attentions: bool = True):
    if not any(model_dir.iterdir()):
        config = AutoConfig.from_pretrained(
            pretrain_name, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        model = AutoModel.from_pretrained(pretrain_name, config=config)
        config.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        config = AutoConfig.from_pretrained(
            model_dir, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        model = AutoModel.from_pretrained(model_dir, config=config)
    #for layer in range(12):
    #    model.encoder.layer[layer].apply(model._init_weights)
    return model


def get_hgfs_tokenizer(pretrain_name: str, tokenizer_dir: Path):
    if not any(tokenizer_dir.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
        tokenizer.save_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer


