'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json

import torch
from torch.autograd import Variable
from transformers import BertTokenizer

from .model_base import modelClassificationBase

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class modelClassificationBaseBert(modelClassificationBase):
    '''
    Classfication.
    Rewrite vocabulary module.
    Use Bert encoder.
    '''

    def __init__(self, args):
        super().__init__(args=args)
        self.pretrained_models = {}

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        pass

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        len_review = []
        review_arr = []
        rating_arr = []
        for line in batch_:
            arr = json.loads(line)
            rating_arr.append(int(arr['label']))

            toks = tokenizer.encode(arr['text'])
            len_review.append(len(toks))
            review_arr.append(toks)

        review_lens = min(self.args.review_max_lens, max(len_review))

        review_arr = [itm[:review_lens] for itm in review_arr]
        review_arr = [itm + [0 for _ in range(review_lens-len(itm))]
                      for itm in review_arr]

        review_var = Variable(torch.LongTensor(review_arr))
        rating_var = Variable(torch.LongTensor(rating_arr))

        pad_mask = Variable(torch.FloatTensor(review_arr))
        pad_mask[pad_mask != float(0)] = -1.0
        pad_mask[pad_mask == float(0)] = 0.0
        pad_mask = -pad_mask

        attn_mask = Variable(torch.FloatTensor(review_arr))
        attn_mask[attn_mask == float(101)] = 0.0
        attn_mask[attn_mask == float(102)] = 0.0
        attn_mask[attn_mask != float(0)] = -1.0
        attn_mask[attn_mask == float(0)] = 0.0
        attn_mask = -attn_mask

        self.batch_data['input_ids'] = review_var.to(self.args.device)
        self.batch_data['label'] = rating_var.to(self.args.device)
        self.batch_data['attn_mask'] = attn_mask.to(self.args.device)
        self.batch_data['pad_mask'] = pad_mask.to(self.args.device)
