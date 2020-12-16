'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import pickle
from pprint import pprint

import numpy as np
import torch

from LeafNATS.data.utils import create_batch_memory
from LeafNATS.engines.end2end_small import End2EndBase
from LeafNATS.utils.utils import show_progress

from .module_attn_vis import createHTML


class End2EndBaseClassification(End2EndBase):
    '''
    End2End training classification.
    Not suitable for language generation task.
    Light weight. Data should be relevatively small.
    '''

    def __init__(self, args=None):
        '''
        Initialize
        '''
        super().__init__(args=args)

    def test_worker(self):
        '''
        Used in decoding.
        Users can define their own decoding process.
        You do not have to worry about path and prepare input.
        '''
        raise NotImplementedError

    def keywords_attnself_worker(self):
        '''
        Keywords worker
        '''
        raise NotImplementedError

    def visualization_attnself_worker(self):
        '''
        visualization worker
        '''
        raise NotImplementedError

    def visualization_attn_abstraction_worker(self):
        '''
        visualization worker
        '''
        raise NotImplementedError

    def keywords_attn_abstraction_worker(self):
        '''
        Keywords worker
        '''
        raise NotImplementedError

    def test(self):
        '''
        Testing
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )

        output_dir = '../nats_results/test'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            self.pred_data = []
            self.true_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()

                self.pred_data += ratePred
                self.true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()
            # save testing data.
            try:
                self.pred_data = np.array(
                    self.pred_data).astype(int)
                np.savetxt(
                    os.path.join(
                        output_dir, '{}_pred_{}.txt'.format(
                            self.args.file_test, self.args.best_epoch)),
                    self.pred_data, fmt='%d')
                self.true_data = np.array(self.true_data).astype(int)
                np.savetxt(
                    os.path.join(
                        output_dir, '{}_true_{}.txt'.format(
                            self.args.file_test, self.args.best_epoch)),
                    self.true_data, fmt='%d')
            except:
                fout = open(os.path.join(
                    output_dir,
                    '{}_pred_{}.pickled'.format(
                        self.args.file_best, self.args.best_epoch)), 'wb')
                pickle.dump(self.pred_data, fout)
                fout.close()
                fout = open(os.path.join(
                    output_dir,
                    '{}_true_{}.pickled'.format(
                        self.args.file_test, self.args.best_epoch)), 'wb')
                pickle.dump(self.true_data, fout)
                fout.close()

    def keywords_attnself(self):
        '''
        Keywords SelfAttention
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )

        output_dir = '../nats_results/keywords_attnself'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            pred_data = []
            true_data = []
            keywords_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()
                output = self.keywords_attnself_worker()
                keywords_data += output

                pred_data += ratePred
                true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()

            for k in range(len(keywords_data)):
                keywords_data[k]['pred_label'] = pred_data[k]
                keywords_data[k]['gold_label'] = true_data[k]

            fout = open(os.path.join(
                output_dir,
                '{}_{}.pickled'.format(
                    self.args.file_test, self.args.best_epoch)), 'wb')
            pickle.dump(keywords_data, fout)
            fout.close()

    def keywords_attn_abstraction(self):
        '''
        Keywords SelfAttention
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower)

        output_dir = '../nats_results/keywords_attn_abstraction'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            pred_data = []
            true_data = []
            keywords_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()
                output = self.keywords_attn_abstraction_worker()
                keywords_data += output

                pred_data += ratePred
                true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()

            for k in range(len(keywords_data)):
                keywords_data[k]['pred_label'] = pred_data[k]
                keywords_data[k]['gold_label'] = true_data[k]

            fout = open(os.path.join(
                output_dir,
                '{}_{}.pickled'.format(
                    self.args.file_test, self.args.best_epoch)), 'wb')
            pickle.dump(keywords_data, fout)
            fout.close()

    def visualization_attnself(self):
        '''
        Keywords SelfAttention
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        output_dir = '../nats_results/visualization_attnself_{}'.format(
            self.args.file_test)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            pred_data = []
            true_data = []
            keywords_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()
                output = self.visualization_attnself_worker()
                keywords_data += output

                pred_data += ratePred
                true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()

            for k in range(len(keywords_data)):
                keywords_data[k]['pred_label'] = pred_data[k]
                keywords_data[k]['gold_label'] = true_data[k]

                file_output = os.path.join(output_dir, '{}.html'.format(k))
                createHTML(keywords_data[k], file_output)
