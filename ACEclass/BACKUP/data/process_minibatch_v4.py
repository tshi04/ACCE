'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json

import torch
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def process_minibatch(input_, max_lens):
    '''
    The data format
    {'text': [], 'label': 0/1}
    '''
    len_review = []
    review_arr = []
    rating_arr = []
    for line in input_:
        arr = json.loads(line)
        rating_arr.append(int(arr['label']))

        toks = tokenizer.encode(arr['text'])
        len_review.append(len(toks))
        review_arr.append(toks)

    review_lens = min(max_lens, max(len_review))

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

    return review_var, rating_var, attn_mask, pad_mask
