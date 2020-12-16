'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from ACEclass.model_base import modelClassificationBase
from LeafNATS.modules.encoder.encoder_cnn import EncoderCNN


class modelClassification(modelClassificationBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.train_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_dim
        ).to(self.args.device)

        kNums = self.args.cnn_kernel_nums.split(',')
        kNums = [int(itm) for itm in kNums]
        ksum = sum(kNums)
        self.train_models['encoder'] = EncoderCNN(
            self.args.emb_dim,
            self.args.cnn_kernel_size,
            self.args.cnn_kernel_nums
        ).to(self.args.device)

        self.train_models['ff'] = torch.nn.Linear(
            ksum, ksum).to(self.args.device)
        self.train_models['classifier'] = torch.nn.Linear(
            ksum, self.args.n_class
        ).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_pipe(self):
        '''
        Shared pipe
        '''
        input_emb = self.train_models['embedding'](
            self.batch_data['input_ids'])
        input_enc = self.train_models['encoder'](input_emb)

        fc = torch.relu(self.train_models['drop'](
            self.train_models['ff'](input_enc)))
        logits = self.train_models['drop'](
            self.train_models['classifier'](fc))

        return logits
