'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from ACEclass.end2end_class import End2EndBaseClassification


class modelClassificationBase(End2EndBaseClassification):
    '''
    Classification, like sentiment analysis.
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        file_ = os.path.join(self.args.data_dir, self.args.file_vocab)
        fp = open(file_, 'r')
        vocab_data = json.load(fp)
        fp.close()
        vocab2id = {}
        id2vocab = {}
        for key in vocab_data:
            vocab2id[vocab_data[key]] = int(key)
            id2vocab[int(key)] = vocab_data[key]
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

    def build_optimizer(self, params):
        '''
        init model optimizer
        '''
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

        return optimizer

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
                vocab2id[wd] if wd in vocab2id else vocab2id['[UNK]']
                for wd in review]
            review_arr.append(review2id)

        review_lens = min(self.args.review_max_lens, max(len_review))

        review_arr = [itm[:review_lens] for itm in review_arr]
        review_arr = [itm + [vocab2id['[PAD]']] *
                      (review_lens-len(itm)) for itm in review_arr]

        review_var = Variable(torch.LongTensor(review_arr))
        rating_var = Variable(torch.LongTensor(rating_arr))

        pad_mask = Variable(torch.FloatTensor(review_arr))
        pad_mask[pad_mask != float(vocab2id['[PAD]'])] = -1.0
        pad_mask[pad_mask == float(vocab2id['[PAD]'])] = 0.0
        pad_mask = -pad_mask

        self.batch_data['input_ids'] = review_var.to(self.args.device)
        self.batch_data['pad_mask'] = pad_mask.to(self.args.device)
        self.batch_data['label'] = rating_var.to(self.args.device)

    def build_pipe(self):
        '''
        Pipes shared by training/validation/testing
        '''
        encoder_output = self.build_encoder()
        attn_output = self.build_attention(encoder_output)
        logits_ = self.build_classifier(attn_output)

        return logits_

    def build_encoder(self):
        '''
        Encoder
        '''
        raise NotImplementedError

    def build_attention(self, input_):
        '''
        Attention
        '''
        raise NotImplementedError

    def build_classifier(self, input_):
        '''
        Classifier
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Data flow from input to output.
        '''
        logits = self.build_pipe()
        logits = logits.contiguous().view(-1, self.args.n_class)
        loss = self.loss_criterion(
            logits, self.batch_data['label'].view(-1))

        return loss

    def test_worker(self):
        '''
        Testing worker
        '''
        logits = self.build_pipe()
        logits = torch.softmax(logits, dim=1)

        ratePred = logits.topk(1, dim=1)[1].squeeze(1).data.cpu().numpy()
        ratePred = ratePred.tolist()

        rateTrue = self.batch_data['label'].data.cpu().numpy()
        rateTrue = rateTrue.tolist()

        return ratePred, rateTrue

    def build_keywords_attnself(self, input_):
        '''
        Keywords
        '''
        raise NotImplementedError

    def build_visualization_attnself(self, input_):
        '''
        visualization
        '''
        raise NotImplementedError

    def keywords_attnself_worker(self):
        '''
        Keywords worker
        '''
        output_enc = self.build_encoder()
        output_attn = self.build_attention(output_enc)
        output_keywords = self.build_keywords_attnself(output_attn)

        return output_keywords

    def visualization_attnself_worker(self):
        '''
        visualization worker
        '''
        output_enc = self.build_encoder()
        output_attn = self.build_attention(output_enc)
        output = self.build_visualization_attnself(output_attn)

        return output

    def build_keywords_attn_abstraction(self, input_):
        '''
        Keywords
        '''
        raise NotImplementedError

    def keywords_attn_abstraction_worker(self):
        '''
        Keywords worker
        '''
        output_enc = self.build_encoder()
        output_attn = self.build_attention(output_enc)
        output_keywords = self.build_keywords_attn_abstraction(output_attn)

        return output_keywords

    def run_evaluation(self):
        '''
        For evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)
        accu = accuracy_score(self.true_data, self.pred_data)
        accu = np.round(accu, 4)

        print('Accuracy={}'.format(accu))
        return accu
