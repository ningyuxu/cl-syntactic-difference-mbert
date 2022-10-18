from typing import List

from datasets.constant import LABEL_OTHERS


class Label:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def num_labels(self) -> int:
        return len(self.get_labels())

    def label_id(self, label) -> int:
        assert label in self.labels, f"Invalid label: {label}"
        return self.label2id[label]

    def map_label(self, label: str) -> str:
        if label in self.labels:
            return label
        if LABEL_OTHERS in self.labels:
            return LABEL_OTHERS
        raise ValueError(f"Invalid label: {label}.")
