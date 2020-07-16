'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import load_vocab_pretrain

from .model_base import modelClassificationBase


class modelClassificationBasePreEmb(modelClassificationBase):
    '''
    Classfication.
    Load pre-trained word embeddings.
    Rewrite vocabulary and base model parameters modules.
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(self.args.data_dir, self.args.file_pretrain_vocab),
            os.path.join(self.args.data_dir, self.args.file_pretrain_vec))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(
            self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)

        for model_name in self.base_models:
            if model_name == 'embedding':
                continue
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        vocab2id = self.batch_data['vocab2id']
        len_review = []
        review_arr = []
        rating_arr = []
        for line in batch_:
            arr = json.loads(line)
            rating_arr.append(int(arr['label']))

            review = arr['text']
            review = list(filter(None, review))
            len_review.append(len(review))

            review2id = [
                vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                for wd in review]
            review_arr.append(review2id)

        review_lens = min(self.args.review_max_lens, max(len_review))

        review_arr = [itm[:review_lens] for itm in review_arr]
        review_arr = [itm + [vocab2id['<pad>']] *
                    (review_lens-len(itm)) for itm in review_arr]

        review_var = Variable(torch.LongTensor(review_arr))
        rating_var = Variable(torch.LongTensor(rating_arr))

        pad_mask = Variable(torch.FloatTensor(review_arr))
        pad_mask[pad_mask != float(vocab2id['<pad>'])] = -1.0
        pad_mask[pad_mask == float(vocab2id['<pad>'])] = 0.0
        pad_mask = -pad_mask

        self.batch_data['input_ids'] = review_var.to(self.args.device)
        self.batch_data['pad_mask'] = pad_mask.to(self.args.device)
        self.batch_data['label'] = rating_var.to(self.args.device)
