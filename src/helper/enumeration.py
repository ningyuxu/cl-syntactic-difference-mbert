from dataclasses import asdict, dataclass


@dataclass
class Enum:
    def choices(self):
        return list(asdict(self).values())


@dataclass
class Split(Enum):
    train: str = "train"
    dev: str = "dev"
    test: str = "test"


@dataclass
class SplitPrefix(Enum):
    train: str = "trn"
    valid: str = "val"
    test: str = "tst"


@dataclass
class Schedule(Enum):
    linear: str = "linear"
    invsqroot: str = "invsqroot"
    reduceOnPlateau: str = "reduceOnPlateau"


@dataclass
class Task(Enum):
    train: str = "train"
    test: str = "test"
    analyse: str = "analyse"


@dataclass
class Model(Enum):
    pretrain: str = "pretrain"
    postagging: str = "postagging"
    parsing: str = "parsing"


@dataclass
class PreTrain(Enum):
    mbert: str = "bert-base-multilingual-cased"


@dataclass
class Corpus(Enum):
    ud_treebanks: str = "ud-treebanks-2.8"
