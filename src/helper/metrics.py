from typing import Dict, List

import torch
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix

from dataloader.constant import LABEL_PADDING_ID


def to_tensor(wrapped_func):
    def func(*args, **kwargs):
        result = wrapped_func(*args, **kwargs)
        return {k: torch.as_tensor(v, dtype=torch.float) for k, v in result.items()}
    return func


class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def unpack(*tensors: torch.Tensor):
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class AccuracyMetric(Metric):
    def __init__(self, num_labels: int):
        self.num_labels = num_labels
        self.gold = []
        self.prediction = []

    def add(self, gold, prediction):
        gold, prediction = self.unpack(gold, prediction)
        _, prediction = torch.max(prediction, dim=1)
        self.gold.extend(gold.tolist())
        self.prediction.extend(prediction.tolist())

    @to_tensor
    def get_metric(self):
        gold = torch.LongTensor(self.gold)
        prediction = torch.LongTensor(self.prediction)
        accuracy = Accuracy()
        acc = accuracy(prediction, gold)
        confmat = ConfusionMatrix(num_classes=self.num_labels)
        confusion_matrix = confmat(prediction, gold)
        return {
            "acc": acc * 100,
            "confusion_matrix": confusion_matrix
        }

    def reset(self):
        self.gold = []
        self.prediction = []


class POSTaggingMetric(Metric):
    def __init__(self, num_labels: int):
        self.num_correct = 0
        self.num_tokens = 0
        self.gold = []
        self.prediction = []
        self.num_labels = num_labels

    def add(self, gold, prediction):
        gold, prediction = self.unpack(gold, prediction)
        _, prediction = torch.max(prediction, dim=-1)
        bs, seq_len = prediction.shape
        for ii in range(bs):
            for jj in range(seq_len):
                gold_label, pred_label = gold[ii, jj], prediction[ii, jj]
                if gold_label == LABEL_PADDING_ID:
                    continue
                self.gold.append(gold_label)
                self.prediction.append(pred_label)
                if gold_label == pred_label:
                    self.num_correct += 1
                self.num_tokens += 1

    @to_tensor
    def get_metric(self):
        try:
            acc = self.num_correct / self.num_tokens
        except ZeroDivisionError:
            acc = 0
        confmat = ConfusionMatrix(num_classes=self.num_labels)
        confusion_matrix = confmat(
            torch.LongTensor(self.prediction), torch.LongTensor(self.gold)
        )
        return {
            "acc": acc * 100,
            "confusion_matrix": confusion_matrix
        }

    def reset(self):
        self.num_correct = 0
        self.num_tokens = 0
        self.gold = []
        self.prediction = []


class ParsingMetric(Metric):
    """
    from allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse, as well as
    sentence level exact match for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None, num_ud_head_labels: int = 38):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
        self._num_ud_head_labels = num_ud_head_labels
        self.pred_head_labels = []
        self.gold_head_labels = []

        self._ignore_classes: List[int] = ignore_classes or []

    def add(
        self,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        mask: torch.Tensor
    ):
        unwrapped = self.unpack(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = unwrapped

        mask = mask.long()
        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask * (~label_mask).long()

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        pred_head_labels = (predicted_labels * correct_indices).tolist()
        gold_head_labels = (gold_labels * correct_indices).tolist()
        self.pred_head_labels.extend([item for sublist in pred_head_labels for item in sublist])
        self.gold_head_labels.extend([item for sublist in gold_head_labels for item in sublist])
        unlabeled_exact_match = (correct_indices + (1 - mask)).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + (1 - mask)).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum().item()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._exact_labeled_correct += labeled_exact_match.sum().item()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (1 - mask).sum().item()

    @to_tensor
    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        if self._total_sentences > 0:
            unlabeled_exact_match = (
                self._exact_unlabeled_correct / self._total_sentences
            )
            labeled_exact_match = self._exact_labeled_correct / self._total_sentences

        assert len(self.pred_head_labels) == len(self.gold_head_labels)
        pred_head_labels = torch.LongTensor(self.pred_head_labels)[torch.LongTensor(self.gold_head_labels).nonzero()]
        gold_head_labels = torch.LongTensor(self.gold_head_labels)[torch.LongTensor(self.gold_head_labels).nonzero()]
        assert pred_head_labels.shape == gold_head_labels.shape, \
            f"shape of pred_head_labels: {pred_head_labels.shape}, shape of gold_head_labels: {gold_head_labels.shape}"
        las_confmat = ConfusionMatrix(num_classes=self._num_ud_head_labels)
        las_confusion_matrix = las_confmat(pred_head_labels, gold_head_labels)

        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
            "uem": unlabeled_exact_match * 100,
            "lem": labeled_exact_match * 100,
            "las_confmat": las_confusion_matrix,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

        self.pred_head_labels = []
        self.gold_head_labels = []


class ParsingPOSMetric(Metric):
    """
    from allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse, as well as
    sentence level exact match for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution itself.
    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None, num_ud_head_labels: int = 38, num_postags: int = 18):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

        self._ignore_classes: List[int] = ignore_classes or []
        self._num_ud_head_labels = num_ud_head_labels
        self.pred_head_labels = []
        self.gold_head_labels = []

        self._num_postags = num_postags
        self.pos_num_correct = 0
        self.pos_num_tokens = 0
        self.pos_gold = []
        self.pos_prediction = []

    def add(
            self,
            gold_indices: torch.Tensor,
            gold_labels: torch.Tensor,
            predicted_indices: torch.Tensor,
            predicted_labels: torch.Tensor,
            mask: torch.Tensor,
            pos_gold,
            pos_prediction
    ):
        unwrapped = self.unpack(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask, pos_gold, pos_prediction
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask, pos_gold, pos_prediction = unwrapped

        mask = mask.long()
        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask * (~label_mask).long()

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        pred_head_labels = (predicted_labels * correct_indices).tolist()
        gold_head_labels = (gold_labels * correct_indices).tolist()
        self.pred_head_labels.extend([item for sublist in pred_head_labels for item in sublist])
        self.gold_head_labels.extend([item for sublist in gold_head_labels for item in sublist])
        unlabeled_exact_match = (correct_indices + (1 - mask)).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + (1 - mask)).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum().item()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._exact_labeled_correct += labeled_exact_match.sum().item()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (1 - mask).sum().item()

        # pos_gold, pos_prediction = self.unpack(pos_gold, pos_prediction)
        _, pos_prediction = torch.max(pos_prediction, dim=-1)
        pos_bs, pos_seq_len = pos_prediction.shape
        for ii in range(pos_bs):
            for jj in range(pos_seq_len):
                pos_gold_label, pos_pred_label = pos_gold[ii, jj], pos_prediction[ii, jj]
                if pos_gold_label == LABEL_PADDING_ID:
                    continue
                self.pos_gold.append(pos_gold_label)
                self.pos_prediction.append(pos_pred_label)
                if pos_gold_label == pos_pred_label:
                    self.pos_num_correct += 1
                self.pos_num_tokens += 1

    @to_tensor
    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        if self._total_sentences > 0:
            unlabeled_exact_match = (
                self._exact_unlabeled_correct / self._total_sentences
            )
            labeled_exact_match = self._exact_labeled_correct / self._total_sentences
        assert len(self.pred_head_labels) == len(self.gold_head_labels)
        pred_head_labels = torch.LongTensor(self.pred_head_labels)[torch.LongTensor(self.gold_head_labels).nonzero()]
        gold_head_labels = torch.LongTensor(self.gold_head_labels)[torch.LongTensor(self.gold_head_labels).nonzero()]
        assert pred_head_labels.shape == gold_head_labels.shape, \
            f"shape of pred_head_labels: {pred_head_labels.shape}, shape of gold_head_labels: {gold_head_labels.shape}"
        las_confmat = ConfusionMatrix(num_classes=self._num_ud_head_labels)
        las_confusion_matrix = las_confmat(pred_head_labels, gold_head_labels)

        try:
            pos_acc = self.pos_num_correct / self.pos_num_tokens
        except ZeroDivisionError:
            pos_acc = 0
        confmat = ConfusionMatrix(num_classes=self._num_postags)
        confusion_matrix = confmat(torch.LongTensor(self.pos_prediction), torch.LongTensor(self.pos_gold))

        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
            "uem": unlabeled_exact_match * 100,
            "lem": labeled_exact_match * 100,
            "pos_acc": pos_acc * 100,
            "confusion_matrix": confusion_matrix,
            "las_confmat": las_confusion_matrix,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

        self.pred_head_labels = []
        self.gold_head_labels = []

        self.pos_num_correct = 0
        self.pos_num_tokens = 0
        self.pos_gold = []
        self.pos_prediction = []
