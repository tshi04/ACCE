'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ACEclass.model_base_bert import modelClassificationBaseBert
from LeafNATS.modules.attention.attention_self import AttentionSelf
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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

        self.train_models['attn_self'] = AttentionSelf(
            input_size=768, hidden_size=768,
            device=self.args.device
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
                self.batch_data['input_ids'],
                self.batch_data['pad_mask'],
                self.batch_data['seg'])[0]

        return input_enc

    def build_attention(self, input_):
        '''
        Attention
        '''
        attn, ctx = self.train_models['attn_self'](
            input_, self.batch_data['att_mask'])

        output = {'attn': attn, 'ctx': ctx}

        return output

    def build_classifier(self, input_):
        '''
        Classifier
        '''
        ctx = input_['ctx']

        fc = torch.relu(self.train_models['drop'](
            self.train_models['ff'](ctx)))
        logits = self.train_models['drop'](
            self.train_models['classifier'](fc))

        return logits

    def build_keywords_attnself(self, input_):
        '''
        Keywords
        '''
        input_ids = self.batch_data['input_ids'].data.cpu().numpy().tolist()
        input_text = []
        for k in range(len(input_ids)):
            out = []
            for j in range(len(input_ids[k])):
                if not input_ids[k][j] == 0:
                    out.append(
                        tokenizer.convert_ids_to_tokens(input_ids[k][j]))
            input_text.append(out)

        attn_ = input_['attn']
        cand_words = attn_.topk(self.args.n_keywords)[
            1].data.cpu().numpy().tolist()
        cand_weights = attn_.topk(self.args.n_keywords)[
            0].data.cpu().numpy().tolist()
        cand_weights = np.around(cand_weights, 4).tolist()

        for k in range(len(cand_words)):
            for j in range(len(cand_words[k])):
                cand_words[k][j] = input_text[k][cand_words[k][j]]

        output = []
        for k in range(len(cand_words)):
            output.append({'toks': cand_words[k], 'weights': cand_weights[k]})

        return output

    def build_visualization_attnself(self, input_):
        '''
        visualization
        '''
        input_weights = input_['attn'].data.cpu().numpy().tolist()
        input_weights = np.around(input_weights, 4).tolist()

        input_ids = self.batch_data['input_ids'].data.cpu().numpy().tolist()
        output_text = []
        output_weights = []
        for k in range(len(input_ids)):
            out_text = []
            out_weight = []
            for j in range(len(input_ids[k])):
                if input_ids[k][j] == 0 or input_ids[k][j] == 101 or input_ids[k][j] == 102:
                    continue
                else:
                    out_text.append(
                        tokenizer.convert_ids_to_tokens(input_ids[k][j]))
                    out_weight.append(input_weights[k][j])
            output_text.append(out_text)
            output_weights.append(out_weight)

        output = []
        for k in range(len(output_text)):
            output.append(
                {'toks': output_text[k], 'weights': output_weights[k]})

        return output
