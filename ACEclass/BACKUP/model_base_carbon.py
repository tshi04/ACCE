'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import load_vocab_pretrain

from .data.process_minibatch_v1 import process_minibatch
from .model_base import modelClassificationBase


class modelClassificationBaseCarbon(modelClassificationBase):
    '''
    Classfication.
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        review, rating, seg, pad_mask, att_mask = process_minibatch(
            batch_, self.args.review_max_lens)
        self.batch_data['input_ids'] = review.to(self.args.device)
        self.batch_data['seg'] = seg.to(self.args.device)
        self.batch_data['label'] = rating.to(self.args.device)
        self.batch_data['pad_mask'] = pad_mask.to(self.args.device)
        self.batch_data['att_mask'] = att_mask.to(self.args.device)
