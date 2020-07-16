'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.modules.attention.attention_self import AttentionSelf


class modelClassification(modelClassificationBaseBertToken):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.train_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_size
        ).to(self.args.device)

        self.train_models['encoderLayer'] = torch.nn.TransformerEncoderLayer(
            self.args.emb_size, self.args.n_heads
        )

        self.train_models['encoder'] = torch.nn.TransformerEncoder(
            self.train_models['encoderLayer'], self.args.n_layers
        ).to(self.args.device)

        self.train_models['attn_self'] = AttentionSelf(
            input_size=self.args.hidden_size,
            hidden_size=self.args.hidden_size*2,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['ff'] = torch.nn.Linear(
            self.args.hidden_size, self.args.hidden_size*2
        ).to(self.args.device)
        self.train_models['classifier'] = torch.nn.Linear(
            self.args.hidden_size*2, self.args.n_class
        ).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_encoder(self):
        '''
        Encoder
        '''
        input_emb = self.train_models['embedding'](
            self.batch_data['input_ids'])
        input_enc = self.train_models['encoder'](input_emb)

        return input_enc

    def build_attention(self, input_):
        '''
        Attention
        '''
        attn, ctx = self.train_models['attn_self'](
            input_, self.batch_data['attn_mask'])

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
                if not input_ids[k][j] == 1:
                    out.append(self.batch_data['id2vocab'][input_ids[k][j]])
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
                if not input_ids[k][j] == 1:
                    out_text.append(
                        self.batch_data['id2vocab'][input_ids[k][j]])
                    out_weight.append(input_weights[k][j])
            output_text.append(out_text)
            output_weights.append(out_weight)

        output = []
        for k in range(len(output_text)):
            output.append(
                {'toks': output_text[k], 'weights': output_weights[k]})

        return output
