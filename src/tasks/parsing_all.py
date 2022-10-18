from typing import Union, Dict, Tuple
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

from helper.parse_tree import decode_mst
from helper import enumeration as enum
from helper.metrics import ParsingMetric
from helper.metrics import ParsingPOSMetric
from models.parsing import ParsingModel
from dataloader.parsing import ParsingDataLoader
from tasks.base import Task


class ParsingTask(Task):
    def __init__(self, hparams: Union[Dict, Namespace]):
        super(ParsingTask, self).__init__(hparams)
        self._comparison = "max"
        self._selection_criterion = "val_las"
        self._metric = ParsingPOSMetric(
            num_ud_head_labels=self.model.num_labels,
            num_postags=self.model.num_pos_tags
        ) if self.hparams.parser_predict_pos else ParsingMetric(num_ud_head_labels=self.model.num_labels)
        self.setup_metrics()

    def define_model(self) -> ParsingModel:
        return ParsingModel(self.hparams)

    def define_dataloader(self) -> ParsingDataLoader:
        return ParsingDataLoader(
            hparams=self.hparams,
            corpus=self.hparams.corpus,
            tokenizer=self.model.tokenizer
        )

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.model.forward(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step_helper(batch, enum.SplitPrefix.valid)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step_helper(batch, enum.SplitPrefix.test)

    def evaluation_step_helper(self, batch, prefix):
        langs = batch[self.model.data_fields[0]]
        token_ids = batch[self.model.data_fields[1]]
        pos_label_ids = batch[self.model.data_fields[2]]
        head_ids = batch[self.model.data_fields[3]]
        arc_label_ids = batch[self.model.data_fields[4]]

        loss, head_tag, child_tag, score_arc, pos_probs = self.model.forward(batch)
        lengths = self.model.pretrain_model.get_mask(token_ids).long().sum(dim=1).cpu().numpy()
        predicted_heads, predicted_labels = self._mst_decode(
            head_tag, child_tag, score_arc, batch["first_subword_mask"], lengths
        )
        evaluation_mask = self._get_mask_for_eval(
            batch["first_subword_mask"], pos_label_ids
        )
        # ignore ROOT evaluation by default as ROOT token is not first subword
        assert (
                len(set(langs)) == 1
        ), "eval batch should contain only one language"
        lang = langs[0]
        if self.hparams.parser_predict_pos:
            self.metrics[lang].add(
                head_ids,
                arc_label_ids,
                predicted_heads,
                predicted_labels,
                evaluation_mask,
                pos_label_ids,
                pos_probs
            )
        else:
            self.metrics[lang].add(
                head_ids,
                arc_label_ids,
                predicted_heads,
                predicted_labels,
                evaluation_mask,
            )

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss
        return result

    def _get_mask_for_eval(
        self, mask: torch.BoolTensor, pos_tags: torch.LongTensor
    ) -> torch.Tensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.
        # Parameters
        mask : `torch.BoolTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.
        # Returns
        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self.model.pos_to_ignore:
            label_mask = pos_tags.eq(label)
            new_mask = new_mask & ~label_mask
        return new_mask.bool()

    def _mst_decode(
            self,
            head_tag: torch.Tensor,
            child_tag: torch.Tensor,
            score_arc: torch.Tensor,
            mask: torch.BoolTensor,
            lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.
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
        # Returns
        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_dim = head_tag.size()

        # lengths = mask.data.sum(dim=1).long().cpu().numpy()
        expanded_shape = [batch_size, sequence_length, sequence_length, tag_dim]
        head_tag = head_tag.unsqueeze(2)
        head_tag = head_tag.expand(*expanded_shape).contiguous()
        child_tag = child_tag.unsqueeze(1)
        child_tag = child_tag.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.model.tag_bilinear(head_tag, child_tag)

        # Note that this log_softmax is over the tag dimension, and we don't consider
        # pairs of tags which are invalid (e.g are a pair which includes a padded
        # element) anyway below. Shape (batch, num_labels,sequence_length,
        # sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(
            pairwise_head_logits, dim=3
        ).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        score_arc = score_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(score_arc, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
            batch_energy: torch.Tensor, lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default
            # it's not necessarily the same in the batched vs un-batched case, which is
            # annoying. Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(np.stack(heads)).to(batch_energy.device),
            torch.from_numpy(np.stack(head_tags)).to(batch_energy.device),
        )
