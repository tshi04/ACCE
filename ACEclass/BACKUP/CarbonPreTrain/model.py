'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ACEclass.model_base_carbon import modelClassificationBaseCarbon
from LeafNATS.modules.embedding.carbon_embedding import CarbonEmbeddings
from LeafNATS.modules.encoder.encoder_carbon import CarbonEncoder
from LeafNATS.modules.attention.attention_self import AttentionSelf


class modelClassification(modelClassificationBaseCarbon):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.base_models['carbon_embedding'] = CarbonEmbeddings(
            self.batch_data['vocab_size'],
            self.args.hidden_size,
            self.args.factor_size,
            0, device=self.args.device
        ).to(self.args.device)

        self.base_models['carbon_encoder'] = CarbonEncoder(
            self.args.hidden_size,
            self.args.factor_size,
            self.args.kernel_size,
            self.args.n_channels,
            self.args.n_layers,
            0, device=self.args.device
        ).to(self.args.device)

        self.train_models['attn_self'] = AttentionSelf(
            input_size=self.args.hidden_size,
            hidden_size=self.args.hidden_size,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['ff'] = torch.nn.Linear(
            self.args.hidden_size, self.args.hidden_size
        ).to(self.args.device)
        self.train_models['classifier'] = torch.nn.Linear(
            self.args.hidden_size, self.args.n_class
        ).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(
            self.args.drop_rate, inplace=True
        ).to(self.args.device)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_pipe(self):
        '''
        Shared pipe
        '''
        with torch.no_grad():
            input_emb = self.base_models['carbon_embedding'](
                self.batch_data['input_ids'])
            input_enc = self.base_models['carbon_encoder'](
                input_emb, self.batch_data['pad_mask'])
            input_enc = input_enc[-1]

        attn, ctx = self.train_models['attn_self'](
            input_enc, self.batch_data['pad_mask'])

        fc = torch.relu(self.train_models['drop'](
            self.train_models['ff'](ctx)))
        logits = self.train_models['classifier'](fc)

        return logits
