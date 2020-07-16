'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel

from ACEclass.model_base_bert2 import modelClassificationBaseBert


class modelClassification(modelClassificationBaseBert):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.pretrained_models['bert'] = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True,
            output_attentions=True
        ).to(self.args.device)

        self.train_models['ff'] = torch.nn.Linear(
            768, 768).to(self.args.device)
        self.train_models['classifier'] = torch.nn.Linear(
            768, self.args.n_class).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_encoder(self):
        '''
        Encoder
        '''
        with torch.no_grad():
            input_enc = self.pretrained_models['bert'](
                self.batch_data['input_ids'], self.batch_data['pad_mask'])[0]
            input_enc = input_enc[:,0,:]

        return input_enc

    def build_attention(self, input_):
        '''
        Attention
        '''
        return input_

    def build_classifier(self, input_):
        '''
        Classifier
        '''
        fc = torch.relu(self.train_models['drop'](
            self.train_models['ff'](input_)))
        logits = self.train_models['drop'](
            self.train_models['classifier'](fc))

        return logits
