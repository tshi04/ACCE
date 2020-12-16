'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
from ACEclass.model_base_bert import modelClassificationBaseBert
from LeafNATS.modules.attention.attention_concepts import Attention_Concepts
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

        self.train_models['attn_concept'] = Attention_Concepts(
            input_size=768,
            n_concepts=self.args.n_concepts,
            device=self.args.device
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
        attn_cpt, ctx_cpt = self.train_models['attn_concept'](
            input_, self.batch_data['att_mask'])
        attn, ctx = self.train_models['attn_self'](ctx_cpt)

        output = {
            'attn': attn, 'ctx': ctx,
            'attn_concept': attn_cpt, 'ctx_concept': ctx_cpt}

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

        attn_cpt = input_['attn_concept']
        batch_size = attn_cpt.size(0)
        cpt_cross = torch.bmm(attn_cpt, attn_cpt.transpose(1, 2))
        diag = torch.eye(
            cpt_cross.size(1), cpt_cross.size(2)
        ).to(self.args.device)
        diag = diag.unsqueeze(0).repeat(batch_size, 1, 1)
        cpt_cross = cpt_cross - diag

        return logits, cpt_cross

    def build_pipe(self):
        '''
        Pipes shared by training/validation/testing
        '''
        encoder_output = self.build_encoder()
        attn_output = self.build_attention(encoder_output)
        logits_, cpt_cross = self.build_classifier(attn_output)

        return logits_, cpt_cross

    def build_pipelines(self):
        '''
        Data flow from input to output.
        '''
        logits, cpt_cross = self.build_pipe()
        logits = logits.contiguous().view(-1, self.args.n_class)

        loss = self.loss_criterion(
            logits, self.batch_data['label'].view(-1))
        loss_cross = torch.sqrt(torch.mean(cpt_cross*cpt_cross))

        return loss + loss_cross

    def test_worker(self):
        '''
        Testing.
        '''
        logits, _ = self.build_pipe()
        prob = torch.softmax(logits, dim=1)

        ratePred = prob.topk(1, dim=1)[1].squeeze(1).data.cpu().numpy()
        ratePred = ratePred.tolist()

        rateTrue = self.batch_data['label'].data.cpu().numpy()
        rateTrue = rateTrue.tolist()

        return ratePred, rateTrue

    def build_keywords_attn_abstraction(self, input_):
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

        attn_ = input_['attn'].data.cpu().numpy().tolist()
        attn_ = np.around(attn_, 4).tolist()
        attn_abs = input_['attn_concept']

        cand_words = attn_abs.topk(self.args.n_keywords)[1]
        cand_words = cand_words.data.cpu().numpy().tolist()
        cand_weights = attn_abs.topk(self.args.n_keywords)[0]
        cand_weights = cand_weights.data.cpu().numpy().tolist()
        cand_weights = np.around(cand_weights, 4).tolist()

        for k in range(len(cand_words)):
            for j in range(len(cand_words[k])):
                for i in range(len(cand_words[k][j])):
                    cand_words[k][j][i] = input_text[k][cand_words[k][j][i]]

        output = []
        for k in range(len(cand_words)):
            output.append({
                'toks': cand_words[k],
                'weights': cand_weights[k],
                'weights_concepts': attn_[k]})

        return output

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

        attn_self = input_['attn']
        attn_concept = input_['attn_concept']
        attn_ = attn_self.unsqueeze(1) @ attn_concept
        attn_ = attn_.squeeze(1)
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
        attn_self = input_['attn']
        attn_concept = input_['attn_concept']
        attn_ = attn_self.unsqueeze(1) @ attn_concept
        attn_ = attn_.squeeze(1)
        input_weights = attn_.data.cpu().numpy().tolist()
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
