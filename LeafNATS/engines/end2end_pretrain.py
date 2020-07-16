'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import json
import os
import pickle
import re
import shutil
import time
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_file
from LeafNATS.utils.utils import show_progress


class End2EndBase(object):
    '''
    This engine is for the end2end training for pretraining.
    '''

    def __init__(self, args=None):
        '''
        Initialize
        '''
        self.args = args
        self.train_models = {}
        self.batch_data = {}
        self.global_steps = 0

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        raise NotImplementedError

    def build_models(self):
        '''
        self.train_models: models that will be trained.
            Format: {'name1': model1, 'name2': model2}
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Pipelines and loss here.
        '''
        raise NotImplementedError

    def build_optimizer(self, params):
        '''
        define optimizer
        '''
        raise NotImplementedError

    def build_batch(self, batch_id):
        '''
        process batch data.
        '''
        raise NotImplementedError

    def train(self):
        '''
        training here.
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        pprint(self.train_models)
        # here it is necessary to put list.
        for model_name in self.train_models:
            try:
                params += list(self.train_models[model_name].parameters())
            except:
                params = list(self.train_models[model_name].parameters())
        print('Total number of parameters: {}.'.format(
            sum([para.numel() for para in params])))
        # define optimizer
        optimizer = self.build_optimizer(params)
        # load checkpoint
        uf_epoch = 0
        out_dir = os.path.join('..', 'nats_results')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if self.args.continue_training:
            model_para_files = glob.glob(os.path.join(out_dir, '*.model'))
            if len(model_para_files) > 0:
                for fl_ in model_para_files:
                    arr = re.split(r'\/', fl_)[-1]
                    arr = re.split(r'\_|\.', arr)
                    epoch_idx = int(arr[-2])
                    if epoch_idx > uf_epoch:
                        uf_epoch = epoch_idx
                try:
                    print("Try *_{}.model".format(uf_epoch))
                    for model_name in self.train_models:
                        fl_ = os.path.join(
                            out_dir, '{}_{}.model'.format(model_name, uf_epoch))
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_, map_location=lambda storage, loc: storage))
                except:
                    uf_epoch -= 1
                    if uf_epoch == -1:
                        uf_epoch = 0
                    else:
                        print("Try *_{}.model".format(uf_epoch))
                        for model_name in self.train_models:
                            fl_ = os.path.join(
                                out_dir, '{}_{}.model'.format(model_name, uf_epoch))
                            self.train_models[model_name].load_state_dict(
                                torch.load(fl_, map_location=lambda storage, loc: storage))

                print('Continue training with *_{}.model'.format(uf_epoch))
        else:
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)
        # train models
        fout = open('../nats_results/args.pickled', 'wb')
        pickle.dump(self.args, fout)
        fout.close()
        start_time = time.time()
        cclb = 0
        if uf_epoch < 0:
            uf_epoch = 0
        for epoch in range(uf_epoch, self.args.n_epoch):
            n_batch = create_batch_file(
                path_data=self.args.data_dir,
                path_work=os.path.join('..', 'nats_results'),
                is_shuffle=True,
                fkey_=self.args.task,
                file_=self.args.file_train,
                batch_size=self.args.batch_size,
                is_lower=self.args.is_lower)
            print('The number of batches: {}'.format(n_batch))
            for batch_id in range(n_batch):
                
                self.build_batch(batch_id)
                loss = self.build_pipelines()

                if loss != loss:
                    raise ValueError

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)
                optimizer.step()

                end_time = time.time()
                if batch_id % self.args.checkpoint == 0:
                    for model_name in self.train_models:
                        fmodel = open(os.path.join(
                            out_dir, model_name+'_'+str(epoch)+'_'+str(batch_id)+'.model'), 'wb')
                        torch.save(
                            self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                    if not os.path.exists(os.path.join(out_dir, 'model')):
                        os.mkdir(os.path.join(out_dir, 'model'))
                    for model_name in self.train_models:
                        fmodel = open(os.path.join(
                            out_dir, 'model', model_name+'.model'), 'wb')
                        torch.save(
                            self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                if batch_id % 1 == 0:
                    end_time = time.time()
                    print('epoch={}, batch={}, loss={}, time_escape={}s={}h'.format(
                        epoch, batch_id,
                        np.round(float(loss.data.cpu().numpy()), 4),
                        np.round(end_time-start_time, 2),
                        np.round((end_time-start_time)/3600.0, 4)))
                del loss
