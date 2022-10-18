from tqdm import tqdm
from typing import List, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from datasets.constant import ISO638_TO_LANG


class Dataset(TorchDataset):
    """
    Dataset classes inherited from pytorch Dataset class. Load data
    from corpus files, and format data for deep learning models.
    """
    def __init__(
            self,
            *,
            tokenizer: PreTrainedTokenizer,
            lang: str,
            split: Optional[str] = None,
            max_len: Optional[int] = None,
            subset_ratio: float = 1,
            subset_count: int = -1,
            subset_seed: int = 25,
    ):
        super(Dataset, self).__init__()

        self.tokenizer = tokenizer
        self.lang = self.unpack_language(lang)
        self.split = split

        model_max_length = self.tokenizer.model_max_length
        if max_len is None:
            max_len = model_max_length
        assert 0 < max_len <= model_max_length, f"Invalid parameter: max_len {max_len}."
        self.max_len = max_len

        assert 0 < subset_ratio <= 1, f"Invalid parameter: subset_ration {subset_ratio}."
        assert not (
            subset_ratio < 1 and subset_count > 0
        ), "Invalid parameter: subset_ratio and subset_count is mutually exclusive."
        self.subset_ratio = subset_ratio
        self.subset_count = subset_count
        self.subset_seed = subset_seed

        self.data: List[Dict[str, np.ndarray]] = []
        self._fields_mapping = dict()

        self.before_load()
        self.load_data()

    @property
    def data_path(self) -> str:
        """
        Directory in which corpus data located.
        """
        raise NotImplementedError

    @property
    def corpus_fields(self) -> List[str]:
        """
        List of corpus data fields' name.
        """
        raise NotImplementedError

    @property
    def data_fields(self) -> List[str]:
        """
        List of self.data fields' name.
        """
        raise NotImplementedError

    @property
    def fields_mapping(self) -> Dict:
        """
        List of API data fields' name.
        """
        return self._fields_mapping

    @fields_mapping.setter
    def fields_mapping(self, values: Dict):
        """
        Define API data format.
        """
        self._fields_mapping = dict()
        for k in values:
            if k in self.data_fields:
                self._fields_mapping[k] = values[k]

    def before_load(self):
        """
        This method will be called before load data, can do some init work.
        """
        pass

    def get_file(self, iso638_lang: str, split: str) -> Optional[str]:
        """
        Get data file from specific path for specific language and split.

        Parameters
        ----------
        iso638_lang: language code (ISO638 standard)
        split: dataset split, such as ``train``/``test``

        Returns
        -------
        data file name
        """
        raise NotImplementedError

    def read_file(self, filepath: str) -> Iterator[Dict]:
        """
        Read data from dataset file.

        Parameters
        ----------
        filepath: data filename

        Returns
        -------
        data iteration, each data item is a key value pair.
        """
        raise NotImplementedError

    def process_example(self, example: Dict) -> List[Dict]:
        """
        Parse data and format data.

        Parameters
        ----------
        example: data item in key value pair format

        Returns
        -------
        tokens and information about tokens.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        data = dict()
        for key_from, key_to in self.fields_mapping.items():
            data[key_to] = self.data[index][key_from]
        return data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def unpack_language(lang: str) -> str:
        assert lang in ISO638_TO_LANG, f"Unsupported language {lang}."
        return lang

    def load_data(self):
        """
        Skeleton of dataset loader, call read_file method to load data.
        filter dataset and shuffle dataset.
        call process_example method to parse data to tokens.

        Returns
        -------
        assign ready to use data to self.data
        """
        assert self.data == []

        examples = []
        filepath = self.get_file(self.lang, self.split)

        for ex in tqdm(self.read_file(filepath), desc="read data"):
            examples.append(ex)

        if self.subset_count > 0 or self.subset_ratio < 1:
            if self.subset_count > 0:
                subset_size = self.subset_count
            elif self.subset_ratio < 1:
                subset_size = int(len(examples) * self.subset_ratio)
            else:
                raise ValueError(
                    "subset_ratio and subset_count is mutually exclusive"
                )

            print(
                f"taking {subset_size} subset "
                f"(total {len(examples)}) "
                f"from {filepath}"
            )

            seed = np.random.RandomState(self.subset_seed)
            examples = seed.permutation(examples)[:subset_size]

        data = []
        for example in tqdm(examples, desc="parse data"):
            data.extend(self.process_example(example))
        self.data = data
