import os
from pathlib import Path
from typing import Union, Dict, List, Optional
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import AutoModel
from transformers import BertModel
from transformers import RobertaModel
from transformers import XLMModel

from helper import utils
from helper import modules
from datasets.constant import ISO638_TO_LANG
from models.constant import HGFS_PATH


class XBertModel(nn.Module):
    def __init__(
            self,
            hparams: Union[Dict, Namespace],
            tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super(XBertModel, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.freeze_layer = hparams.freeze_layer
        self.weighted_feature = hparams.weighted_feature
        self.feature_layer = hparams.feature_layer
        self.projector = hparams.projector
        self.projector_trm_hidden_size = hparams.projector_trm_hidden_size
        self.projector_trm_num_heads = hparams.projector_trm_num_heads
        self.projector_dropout = hparams.projector_dropout
        self.projector_trm_num_layers = hparams.projector_trm_num_layers

        self._tokenizer = tokenizer
        self.model = self.define_model()
        self.freeze_layers()

        self.weight = nn.Parameter(torch.zeros(self.num_layers))
        self.mapping = None
        if hparams.mapping:
            assert os.path.isfile(hparams.mapping)
            self.mapping = torch.load(hparams.mapping)
            self.freeze_module(self.mapping)
        self.projector = self.build_projector()
        self.dropout = modules.InputVariationalDropout(hparams.input_dropout)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = utils.get_hgfs_tokenizer(self.hparams.pretrain, Path(HGFS_PATH + "/tokenizer"))
        return self._tokenizer

    def define_model(self):
        pretrain_model = utils.get_hgfs_model(self.hparams.pretrain, Path(HGFS_PATH + "/model"))
        return pretrain_model

    def model_output(
            self,
            sent: Tensor,
            langs: Optional[List[str]] = None,
            segment: Optional[Tensor] = None
    ):
        mask = self.get_mask(sent)
        if isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel):
            output = self.model(
                input_ids=sent, attention_mask=mask, token_type_ids=segment
            )
        elif isinstance(self.model, XLMModel):
            lang_ids: Optional[torch.Tensor] = None
            if langs is not None:
                try:
                    batch_size, seq_len = sent.shape
                    lang_ids = torch.tensor(
                        [self.tokenizer.lang2id[lang] for lang in langs],
                        dtype=torch.long, device=sent.device,
                    )
                    lang_ids = lang_ids.unsqueeze(1).expand(batch_size, seq_len)
                except KeyError as e:
                    print(f"KeyError with missing language {e}")
                    lang_ids = None
            output = self.model(
                input_ids=sent, attention_mask=mask, langs=lang_ids, token_type_ids=segment
            )
        else:
            raise ValueError("Unsupported model.")
        return output

    def encode_sent(
            self,
            sent: Tensor,
            langs: Optional[List[str]] = None,
            segment: Optional[Tensor] = None
    ):
        output = self.model_output(sent, langs, segment)
        hidden_states = output["hidden_states"]
        hs = self.map_feature(hidden_states, langs)
        hs = self.process_feature(hs)
        hs = self.dropout(hs)
        mask = self.get_mask(sent)
        sent_embedding = self.projector(hs, mask)
        return sent_embedding

    def map_feature(self, hidden_states: List[Tensor], langs):
        if self.mapping is None:
            return hidden_states
        assert len(set(langs)) == 1, "a batch should contain only one language"
        lang = langs[0]
        assert lang in ISO638_TO_LANG, "Unsupported language."
        if lang not in self.mapping:
            return hidden_states
        hs = []
        for h, m in zip(hidden_states, self.mapping[lang]):
            hs.append(m(h))
        return hs

    def process_feature(self, hidden_states: List[Tensor]):
        if self.weighted_feature:
            hs: Tensor = torch.stack(hidden_states)
            weight = F.softmax(self.weight, dim=0).view(-1, 1, 1, 1)
            hs = hs * weight
            hs = hs.sum(dim=0)
        else:
            hs = hidden_states[self.feature_layer]
        return hs

    def build_projector(self):
        if self.projector == "identity":
            return modules.Identity()
        elif self.projector == "meanpool":
            return modules.MeanPooling()
        elif self.projector == "transformer":
            return modules.Transformer(
                input_dim=self.hidden_size,
                hidden_dim=self.projector_trm_hidden_size,
                num_heads=self.projector_trm_num_heads,
                dropout=self.projector_dropout,
                num_layers=self.projector_trm_num_layers,
            )
        else:
            raise ValueError(self.projector)

    @property
    def hidden_size(self):
        if isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel):
            return self.model.config.hidden_size
        elif isinstance(self.model, XLMModel):
            return self.model.dim
        else:
            raise ValueError("Unsupported model")

    @property
    def num_layers(self):
        if isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel):
            return self.model.config.num_hidden_layers + 1
        elif isinstance(self.model, XLMModel):
            return self.model.n_layers + 1
        else:
            raise ValueError("Unsupported model")

    @property
    def attention_heads(self):
        if isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel):
            return self.model.config.num_attention_heads
        elif isinstance(self.model, XLMModel):
            return self.model.n_heads
        else:
            raise ValueError("Unsupported model")

    def get_mask(self, sent: Tensor):
        mask = (sent != self.tokenizer.pad_token_id).long()
        return mask

    @classmethod
    def freeze_module(cls, module):
        for param in module.parameters():
            param.requires_grad = False

    def freeze_layer(self, layer):
        if isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel):
            self.freeze_module(self.model.encoder.layer[layer - 1])
        elif isinstance(self.model, XLMModel):
            self.freeze_module(self.model.attentions[layer - 1])
            self.freeze_module(self.model.layer_norm1[layer - 1])
            self.freeze_module(self.model.ffns[layer - 1])
            self.freeze_module(self.model.layer_norm2[layer - 1])
        else:
            raise ValueError("Unsupported model")

    def freeze_embeddings(self):
        if isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel):
            self.freeze_module(self.model.embeddings)
        elif isinstance(self.model, XLMModel):
            self.freeze_module(self.model.position_embeddings)
            if self.model.n_langs > 1 and self.model.use_lang_emb:
                self.freeze_module(self.model.lang_embeddings)
            self.freeze_module(self.model.embeddings)
        else:
            raise ValueError("Unsupported model")

    def freeze_layers(self):
        if self.freeze_layer == -1:
            return
        elif self.freeze_layer >= 0:
            for i in range(self.freeze_layer + 1):
                if i == 0:
                    print("freeze embeddings")
                    self.freeze_embeddings()
                else:
                    print(f"freeze layer {i}")
                    self.freeze_layer(i)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--freeze_layer", default=-1, type=int)
        parser.add_argument("--feature_layer", default=-1, type=int)
        parser.add_argument("--weighted_feature", default=False, type=utils.str2bool)
        parser.add_argument(
            "--projector", default="identity",
            choices=["identity", "meanpool", "transformer"], type=str
        )
        parser.add_argument("--projector_trm_hidden_size", default=3072, type=int)
        parser.add_argument("--projector_trm_num_heads", default=12, type=int)
        parser.add_argument("--projector_trm_num_layers", default=4, type=int)
        parser.add_argument("--projector_dropout", default=0.2, type=float)
        parser.add_argument("--input_dropout", default=0.2, type=float)
        parser.add_argument("--mapping", default="", type=str)
        return parser
