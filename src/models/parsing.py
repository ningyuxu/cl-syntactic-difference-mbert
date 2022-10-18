from typing import Union, Dict, List, Tuple
from argparse import Namespace, ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F

from helper import utils
from helper.modules import BiLinearAttention, InputVariationalDropout
from datasets.constant import UD_POS_LABELS
from dataloader.constant import LABEL_PADDING_ID
from dataloader.parsing import ParsingDataLoader
from models.base import Model


class ParsingModel(Model):

    DATA_FIELDS = ["lang", "token_ids", "pos_label_ids", "head_ids", "arc_label_ids"]

    POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}

    def __init__(self, hparams: Union[Dict, Namespace]):
        super(ParsingModel, self).__init__(hparams)

        self.num_labels = len(self.hparams.labels)
        self.num_pos_tags = len(UD_POS_LABELS)

        encode_dim = self.pretrain_model.hidden_size
        if hparams.parser_use_pos or hparams.parser_use_predict_pos:
            if hparams.parser_use_pos:
                num_pos_tags = self.num_pos_tags + 1
                padding_idx = LABEL_PADDING_ID
            elif hparams.parser_use_predict_pos:
                assert (
                    hparams.parser_predict_pos
                ), "When parser_use_predict_pos is True, parser_predict_pos should also be True"
                num_pos_tags = self.num_pos_tags
                padding_idx = None
            else:
                raise ValueError(
                    "parser_use_pos and parser_use_predict_pos are mutually exclusive"
                )
            self.pos_embed = nn.Embedding(
                num_pos_tags,
                hparams.parser_pos_dim,
                padding_idx=padding_idx
            )
            encode_dim += hparams.parser_pos_dim

        if hparams.parser_predict_pos:
            self.pos_tagger = nn.Linear(self.pretrain_model.hidden_size, self.num_pos_tags)

        self.head_arc_ff = self._ff(
            encode_dim, hparams.parser_arc_dim, hparams.parser_dropout
        )
        self.child_arc_ff = self._ff(
            encode_dim, hparams.parser_arc_dim, hparams.parser_dropout
        )
        self.arc_attention = BiLinearAttention(
            hparams.parser_arc_dim, hparams.parser_arc_dim, use_input_biases=True
        )
        self.head_tag_ff = self._ff(
            encode_dim, hparams.parser_tag_dim, hparams.parser_dropout
        )
        self.child_tag_ff = self._ff(
            encode_dim, hparams.parser_tag_dim, hparams.parser_dropout
        )
        self.tag_bilinear = nn.Bilinear(
            hparams.parser_tag_dim, hparams.parser_tag_dim, self.num_labels
        )

        punctuation_tag_indices = {
            pos_tag: index
            for index, pos_tag in enumerate(UD_POS_LABELS)
            if pos_tag in self.POS_TO_IGNORE
        }
        self.pos_to_ignore = set(punctuation_tag_indices.values())
        print(
            f"Found POS tags corresponding to the following punctuation : "
            f"{punctuation_tag_indices}, ignoring words with these POS tags for "
            f"evaluation."
        )

    @property
    def data_fields(self) -> List[str]:
        return ParsingDataLoader.DATA_FIELDS

    @classmethod
    def _ff(cls, input_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU(),
            InputVariationalDropout(dropout),
        )

    def preprocess_batch(self, x):
        _, seq_len = x[self.data_fields[1]].shape
        head_ids = x[self.data_fields[3]]
        x["first_subword_mask"] = head_ids != -1
        x[self.data_fields[3]] = head_ids.masked_fill(head_ids >= seq_len, -1)
        return x

    def forward(self, x):
        x = self.preprocess_batch(x)
        lang = x[self.data_fields[0]]
        token_ids = x[self.data_fields[1]]
        pos_label_ids = x[self.data_fields[2]]
        head_ids = x[self.data_fields[3]]
        arc_label_ids = x[self.data_fields[4]]
        first_subword_mask = x["first_subword_mask"]

        hs = self.pretrain_model.encode_sent(sent=token_ids, langs=lang)
        if self.hparams.parser_predict_pos:
            logits = self.pos_tagger(hs)
            log_probs = F.log_softmax(logits, dim=-1)
            pos_nll = F.nll_loss(
                log_probs.view(-1, self.num_pos_tags),
                x["pos_label_ids"].view(-1),
                ignore_index=LABEL_PADDING_ID,
            )
        else:
            log_probs = None
            pos_nll = 0

        if self.hparams.parser_use_pos:
            hs_pos = self.pos_embed(
                pos_label_ids.masked_fill(pos_label_ids < 0, self.num_pos_tags)
            )
            hs = torch.cat((hs, hs_pos), dim=-1)
        elif self.hparams.parser_use_predict_pos:
            assert log_probs is not None
            hs_pos = F.linear(log_probs.exp().detach(), self.pos_embed.weight.t())
            hs = torch.cat((hs, hs_pos), dim=-1)

        head_arc = self.head_arc_ff(hs)
        child_arc = self.child_arc_ff(hs)
        score_arc = self.arc_attention(head_arc, child_arc)

        head_tag = self.head_tag_ff(hs)
        child_tag = self.child_tag_ff(hs)

        minus_inf = -1e8
        minus_mask = ~first_subword_mask * minus_inf
        score_arc = score_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        arc_nll, tag_nll = self._construct_loss(
            head_tag=head_tag,
            child_tag=child_tag,
            score_arc=score_arc,
            head_indices=head_ids,
            head_tags=arc_label_ids,
            mask=first_subword_mask,
        )
        loss = arc_nll + tag_nll + pos_nll

        return loss, head_tag, child_tag, score_arc, log_probs

    def _construct_loss(
        self,
        head_tag: torch.Tensor,
        child_tag: torch.Tensor,
        score_arc: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.
        # Parameters
        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        score_arc : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to
            generate a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting un-padded
            elements in the sequence.
        # Returns
        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = score_arc.size()
        # shape (batch_size, 1)
        range_vector = torch.arange(batch_size, device=score_arc.device).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
            utils.masked_log_softmax(score_arc, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag, child_tag, head_indices)
        normalised_head_tag_logits = utils.masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = torch.arange(sequence_length, device=score_arc.device)
        child_index = (
            timestep_index.view(1, sequence_length)
            .expand(batch_size, sequence_length)
            .long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _get_head_tags(
            self,
            head_tag: torch.Tensor,
            child_tag: torch.Tensor,
            head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.
        # Parameters
        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.
        # Returns
        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag.size()[0]
        # shape (batch_size, 1)
        range_vector = torch.arange(batch_size, device=head_tag.device).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really need
        # to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word
        # from the sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_dim)
        selected_head_tag = head_tag[range_vector, head_indices]
        selected_head_tag = selected_head_tag.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag, child_tag)
        return head_tag_logits

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--parser_use_pos", default=False, type=utils.str2bool)
        parser.add_argument("--parser_use_predict_pos", default=True, type=utils.str2bool)
        parser.add_argument("--parser_predict_pos", default=True, type=utils.str2bool)
        parser.add_argument("--parser_pos_dim", default=100, type=int)
        parser.add_argument("--parser_tag_dim", default=128, type=int)
        parser.add_argument("--parser_arc_dim", default=512, type=int)
        parser.add_argument("--parser_dropout", default=0.33, type=float)
        return parser
