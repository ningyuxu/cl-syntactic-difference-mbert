import glob
from typing import List, Iterator, Dict, Optional

import numpy as np
from transformers import XLMTokenizer
from transformers import XLMRobertaTokenizer

from helper import enumeration as enum
from datasets.label import Label
from datasets.base import Dataset
from datasets.constant import UD_ISO638_TO_LANG
from datasets.constant import UD_DATA_PATH
from datasets.constant import UD_POS_LABELS
from datasets.constant import UD_HEAD_LABELS
from dataloader.constant import LABEL_PADDING_ID


class UDTreebankDataset(Dataset):
    """
    Dataset class for Universal Dependency Treebank corpus.
    """

    CORPUS_FIELDS = ["tokens", "pos_labels", "heads", "arc_labels"]

    DATA_FIELDS = ["lang", "token_ids", "pos_label_ids", "head_ids", "arc_label_ids"]

    def __init__(
            self,
            *,
            max_len_unit: str,
            pos_labels: Optional[Label] = None,
            arc_labels: Optional[Label] = None,
            **kwargs):
        assert max_len_unit in ["word", "subword"]
        self.max_len_unit = max_len_unit
        self._pos_labels = pos_labels
        self._arc_labels = arc_labels

        super(UDTreebankDataset, self).__init__(**kwargs)

    @property
    def data_path(self) -> str:
        return UD_DATA_PATH

    @property
    def corpus_fields(self) -> List[str]:
        return self.CORPUS_FIELDS

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    @property
    def pos_labels(self) -> Label:
        if not self._pos_labels:
            self._pos_labels = Label(UD_POS_LABELS)
        return self._pos_labels

    @property
    def arc_labels(self) -> Label:
        if not self._arc_labels:
            self._arc_labels = Label(UD_HEAD_LABELS)
        return self._arc_labels

    def before_load(self):
        self.max_len = min(
            self.max_len,
            self.tokenizer.max_len_single_sentence
        )

    def get_file(self, iso638_lang: str, split: str) -> Optional[str]:
        lang = UD_ISO638_TO_LANG[iso638_lang]

        if split == enum.Split.train:
            fp = f"{self.data_path}/UD_{lang}/*-ud-train.conllu"
        elif split == enum.Split.dev:
            fp = f"{self.data_path}/UD_{lang}/*-ud-dev.conllu"
        elif split == enum.Split.test:
            fp = f"{self.data_path}/UD_{lang}/*-ud-test.conllu"
        else:
            raise ValueError(f"Unsupported split: {split}")

        _fp = glob.glob(fp)
        if len(_fp) == 1:
            return _fp[0]
        elif len(_fp) == 0:
            return None
        else:
            raise ValueError(f"Unsupported split: {split}")

    def read_file(self, filepath) -> Iterator[Dict]:
        tokens: List[str] = []
        pos_labels: List[str] = []
        heads: List[int] = []
        arc_labels: List[str] = []
        with open(filepath, mode='r', encoding="utf-8") as f:
            for line in f:
                tok = line.strip().split('\t')
                if len(tok) < 2 or line[0] == "#":
                    if tokens:
                        yield {
                            self.corpus_fields[0]: tokens,
                            self.corpus_fields[1]: pos_labels,
                            self.corpus_fields[2]: heads,
                            self.corpus_fields[3]: arc_labels,
                        }
                        tokens = []
                        pos_labels = []
                        heads = []
                        arc_labels = []
                if tok[0].isdigit():
                    word, head = tok[1], tok[6]
                    pos_label = self.pos_labels.map_label(tok[3])
                    arc_label = tok[7].split(":")[0]
                    arc_label = self.arc_labels.map_label(arc_label)
                    tokens.append(word)
                    pos_labels.append(pos_label)
                    heads.append(int(head))
                    arc_labels.append(arc_label)
            if tokens:
                yield {
                    self.corpus_fields[0]: tokens,
                    self.corpus_fields[1]: pos_labels,
                    self.corpus_fields[2]: heads,
                    self.corpus_fields[3]: arc_labels,
                }

    def process_example(self, example: Dict) -> List[Dict]:
        tokens: List = example[self.corpus_fields[0]]
        pos_labels: List = example[self.corpus_fields[1]]
        heads: List = example[self.corpus_fields[2]]
        arc_labels: List = example[self.corpus_fields[3]]

        token_ids: List[int] = []
        pos_label_ids: List[int] = []
        head_ids: List[int] = []
        arc_label_ids: List[int] = []

        _tokens = [self.tokenize(w) for w in tokens]
        # We use this to convert head position in UD to the first subword position in the
        # tokenized sentence. As UD head position is 1 based, we assume the model prepend
        # *one* special token to a sentence.
        word2subword_pos = np.cumsum([0, 1] + [len(w) for w in _tokens])
        # **max_len in parsing is the number of word instead of subword**
        if self.max_len_unit == "word":
            _tokens = _tokens[: self.max_len]
            max_len = self.tokenizer.max_len_single_sentence
        else:
            max_len = self.max_len
        for sub_tokens, pos_label, head, arc_label in zip(_tokens, pos_labels, heads, arc_labels):
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            if len(token_ids) + len(sub_tokens) >= max_len:
                break
            for i, sub_token in enumerate(sub_tokens):
                token_ids.append(sub_token)
                pos_label_id = self.pos_labels.label_id(pos_label) if i == 0 else LABEL_PADDING_ID
                pos_label_ids.append(pos_label_id)

                assert head >= 0, "head < 0"
                head_id = word2subword_pos[head] if i == 0 else -1
                head_ids.append(head_id)

                arc_label_id = self.arc_labels.label_id(arc_label) if i == 0 else LABEL_PADDING_ID
                arc_label_ids.append(arc_label_id)
        token_ids, pos_label_ids, head_ids, arc_label_ids = self.add_special_tokens(
            token_ids, pos_label_ids, head_ids, arc_label_ids
        )
        assert len(token_ids) == len(pos_label_ids) == len(head_ids) == len(arc_label_ids)
        return [
            {
                self.data_fields[0]: self.lang,
                self.data_fields[1]: token_ids,
                self.data_fields[2]: pos_label_ids,
                self.data_fields[3]: head_ids,
                self.data_fields[4]: arc_label_ids,
            }
        ]

    def add_special_tokens(self, sent, pos_tags, heads, labels):
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        pos_tags = self.tokenizer.build_inputs_with_special_tokens(pos_tags)
        heads = self.tokenizer.build_inputs_with_special_tokens(heads)
        labels = self.tokenizer.build_inputs_with_special_tokens(labels)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(sent, already_has_special_tokens=True)
        unk_mask = [1 if t == self.tokenizer.unk_token_id else 0 for t in sent]
        mask = [i ^ j for i, j in zip(special_tokens_mask, unk_mask)]
        sent = np.array(sent)
        pos_tags = np.array(pos_tags)
        heads = np.array(heads)
        labels = np.array(labels)
        mask = np.array(mask)
        pos_tags = pos_tags * (1 - mask) + LABEL_PADDING_ID * mask
        heads = heads * (1 - mask) + -1 * mask
        labels = labels * (1 - mask) + LABEL_PADDING_ID * mask
        return sent, pos_tags, heads, labels

    def tokenize(self, token):
        if isinstance(self.tokenizer, XLMTokenizer):
            sub_words = self.tokenizer.tokenize(token, lang=self.lang)
        else:
            sub_words = self.tokenizer.tokenize(token)
        if isinstance(self.tokenizer, XLMRobertaTokenizer):
            if not sub_words:
                return []
            if sub_words[0] == "▁":
                sub_words = sub_words[1:]
        return sub_words
