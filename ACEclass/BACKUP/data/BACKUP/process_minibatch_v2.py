'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json

import torch
from torch.autograd import Variable


def process_minibatch(input_, vocab2id, max_lens):
    '''
    Process the minibatch for beeradvocate and tripadvisor datasets
    The data format
    {'text': [], 'bert_ids': [], 'label': 0/1}
    '''
    len_review = []
    review_arr = []
    rating_arr = []
    for line in input_:
        arr = json.loads(line)
        rating_arr.append(int(arr['label']))

        review = arr['text']
        review = list(filter(None, review))
        len_review.append(len(review))

        review2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                     for wd in review]
        review_arr.append(review2id)

    review_lens = min(max_lens, max(len_review))

    review_arr = [itm[:review_lens] for itm in review_arr]
    review_arr = [itm + [vocab2id['<pad>'] for _ in range(review_lens-len(itm))]
                  for itm in review_arr]

    review_var = Variable(torch.LongTensor(review_arr))
    rating_var = Variable(torch.LongTensor(rating_arr))

    weight_mask = Variable(torch.FloatTensor(review_arr))
    weight_mask[weight_mask != float(vocab2id['<pad>'])] = -1.0
    weight_mask[weight_mask == float(vocab2id['<pad>'])] = 0.0
    weight_mask = -weight_mask

    return review_var, weight_mask, rating_var
