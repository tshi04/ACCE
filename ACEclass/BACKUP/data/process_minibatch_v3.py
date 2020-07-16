'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json

import torch
from torch.autograd import Variable


def process_minibatch(input_, max_lens):
    '''
    The data format
    {'text': [], 'bert_ids': [], 'label': 0/1}
    '''
    len_review = []
    review_arr = []
    rating_arr = []
    for line in input_:
        arr = json.loads(line)
        rating_arr.append(int(arr['label']))

        len_review.append(len(arr['bert_ids']))
        review_arr.append(arr['bert_ids'])

    review_lens = min(max_lens, max(len_review))

    review_arr = [itm[:review_lens-2] for itm in review_arr]
    review_arr = [[101] + itm + [102] + [0 for _ in range(review_lens-2-len(itm))]
                  for itm in review_arr]
    seg_arr = [[0 for _ in range(review_lens)]
               for k in range(len(review_arr))]

    review_var = Variable(torch.LongTensor(review_arr))
    rating_var = Variable(torch.LongTensor(rating_arr))
    seg_var = Variable(torch.LongTensor(seg_arr))

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

    return review_var, rating_var, seg_var, pad_mask, attn_mask
