from typing import Union, Dict, List
from argparse import Namespace

from torch import nn
from torch.nn import functional as F

from dataloader.constant import LABEL_PADDING_ID
from dataloader.postagging import POSTaggingDataLoader
from models.base import Model


class POSTaggingModel(Model):

    def __init__(self, hparams: Union[Dict, Namespace]):
        super(POSTaggingModel, self).__init__(hparams)

        self.num_labels = len(self.hparams.labels)
        self.classifier = nn.Linear(self.pretrain_model.hidden_size, self.num_labels)

    @property
    def data_fields(self) -> List[str]:
        return POSTaggingDataLoader.DATA_FIELDS

    def forward(self, x):
        lang = x[self.data_fields[0]]
        token_ids = x[self.data_fields[1]]
        emb = self.pretrain_model.encode_sent(sent=token_ids, langs=lang)
        logits = self.classifier(emb)
        log_probs = F.log_softmax(logits, dim=2)
        loss = F.nll_loss(
            input=log_probs.view(-1, self.num_labels),
            target=x[self.data_fields[2]].long().view(-1),
            ignore_index=LABEL_PADDING_ID
        )
        return loss, log_probs
