
import os
import json
import pickle
import torch
import torch.nn
from typing import List, Dict
from codes.supports.utils import *
import yaml

with open('../config.yaml') as f:
    config = yaml.safe_load(f)['exp']

class Storer:
    """
    Class to store data of one model.
    """

    def __init__(self, save_dir: str, lowest_dir: str, hla_list: List) -> None:
        """
        Args:
            save_dir (str) : 
        Returns:
            None
        """
        self.lowest_dir = lowest_dir
        self.hla_list = hla_list
        self.model_save_dir = save_dir + config['model_save_loc'] + lowest_dir
        #self.csv_save_dir = save_dir + config['csv_save_loc']
        self.log_save_dir = save_dir + config['log_save_loc'] + lowest_dir

        makedirs(self.model_save_dir)
        #makedirs(self.csv_save_dir)
        makedirs(self.log_save_dir)

        self.trains = {}
        self.evals = {}

        for hla in hla_list:
            self.trains[hla] = {'acc': {}}
            self.evals[hla] = {'acc': {}}

    def store_epoch_result(self, epoch: int, acc_dict: Dict, is_eval: bool) -> None:
        """
        Args:
            epoch (int):
            loss_dict (Dict): Expected to be {'HLA_hoge':0.0, 'HLA_fuga':0.1}
            acc_dict (Dict): Expected to be {'HLA_hoge':1.0, 'HLA_fuga':0.99}
            is_eval (bool):
        Returns:
            None
        """
        for hla in self.hla_list:
            if is_eval:
                self.evals[hla]['acc'][epoch] = acc_dict[hla]
            else:
                self.trains[hla]['acc'][epoch] = acc_dict[hla]

    def amend_model(self, modeldict: Dict, is_best: bool) -> None:
        """
        Amend model if criteria score is higher than ever.
        Args:
            model_dict (Dict): Expected to be {'shared':nn.Module, 'HLA_hoge':nn.Module}
            is_best (bool) :
        Returns:
            None
        """

        if is_best:
            save_name = self.model_save_dir  + '/shared.pth'
            torch.save(modeldict['shared'].state_dict(), save_name)
            print('Model is amended.')
            
            for hla in self.hla_list:
                save_name = self.model_save_dir +  f'/{hla}.pth'
                torch.save(modeldict[hla].state_dict(), save_name)

    def save_logs(self) -> None:
        """
        Dump acc and loss of each epoch.
        Args:
            None
        Returns:
            None
        """
        with open(self.log_save_dir +  '/train_scores.json', 'w') as ft:
            json.dump(self.trains, ft, indent=4)

        with open(self.log_save_dir + '/eval_scores.json', 'w') as fe:
            json.dump(self.evals, fe, indent=4)

class AllResultStorer:

    def __init__(self, save_dir: str, is_test=False) -> None:
        """
        Args:
            save_dir (str) : Path to save results
        Returns:
            None
        """
        self.is_test = is_test
        self.csv_save_dir = save_dir + config['csv_save_loc']
        makedirs(self.csv_save_dir)

        self.model_info_save_dir = save_dir + config['model_save_loc']
        if not is_test:
            self.result_dict_t = {'hla':[], 'digit':[], 'r2':[], 'r2_01':[], 'concordance':[], 'ppv':[], 'sens':[], 'fscore':[], 'freq':[], 'confidence':[]}
            self.result_dict_v = {'hla':[], 'digit':[], 'r2':[], 'r2_01':[],'concordance':[], 'ppv':[], 'sens':[], 'fscore':[], 'freq':[], 'confidence':[]}
            self.sample_dict_t = {'hla':[], 'digit':[], 'R2':[],'accuracy':[], 'freq':[]}
            self.sample_dict_v = {'hla':[], 'digit':[], 'R2':[], 'accuracy':[], 'freq':[]}
            self.model_info_dict = {}#{1(idx):{input_len:25, hla_list:{2-digit:[], 4-digit:[]}} }
        elif is_test:
            self.result_dict = {'hla':[], 'digit':[], 'r2':[], 'r2_01':[], 'concordance':[],'ppv':[], 'sens':[], 'fscore':[], 'freq':[], 'confidence':[]}
            self.sample_dict = {'hla':[], 'digit':[], 'R2':[], 'accuracy':[], 'freq':[]}

    def store_hla_list(self, idx, digit, hla_list):
        if digit == '2-digit':
            self.model_info_dict[idx]['hla_list'] = {}

        self.model_info_dict[idx]['hla_list'][digit] = hla_list

    def prepare_to_store_model_info(self, idx):
        self.model_info_dict[idx] = {}

    def store_input_len(self, idx, input_len, chunk_len):
        self.model_info_dict[idx]['input_len'] = input_len
        self.model_info_dict[idx]['chunk_len'] = chunk_len

    def store_evals_by_freq(self, result_dict_tmp: Dict, sample_dict_tmp, mode: str) -> None:
        """
        Store ppv, sensitivity, fscore and freq of each allele
        Args:
            kwargs (Dict) : 
        Returns:
            None
        """
        if not self.is_test:
            assert len(result_dict_tmp) == len(self.result_dict_t)
            assert len(result_dict_tmp) == len(self.result_dict_v)
        assert mode  in ['train', 'val', 'test']

        if mode == 'train':
            for key in result_dict_tmp.keys():
                self.result_dict_t[key] += result_dict_tmp[key]
            for key in sample_dict_tmp.keys():
                self.sample_dict_t[key] += sample_dict_tmp[key]

        elif mode == 'val':
            for key in result_dict_tmp.keys():
                self.result_dict_v[key] += result_dict_tmp[key]
            for key in sample_dict_tmp.keys():
                self.sample_dict_v[key] += sample_dict_tmp[key]

        elif mode == 'test':
            for key in result_dict_tmp.keys():
                self.result_dict[key] += result_dict_tmp[key]
                
            for key in sample_dict_tmp.keys():
                self.sample_dict[key] += sample_dict_tmp[key]            

    def dump_results(self) -> None:
        """
        Save ppv, sensitivity, fscore and freq of each allele
        Args:
            None
        Returns;
            None
        """
        if not self.is_test:
            df = pd.DataFrame(self.result_dict_t)
            df.to_csv(self.csv_save_dir + f'/evals_by_freq_train.csv')

            df = pd.DataFrame(self.sample_dict_t)
            df.to_csv(self.csv_save_dir + f'/evals_by_freq_sample_train.csv')

            df = pd.DataFrame(self.result_dict_v)
            df.to_csv(self.csv_save_dir + f'/evals_by_freq_val.csv')

            df = pd.DataFrame(self.sample_dict_v)
            df.to_csv(self.csv_save_dir + f'/evals_by_freq_sample_train.csv')

            model_info_save_loc = self.model_info_save_dir  + '/model_info.json'
            with open(model_info_save_loc, 'w') as fe:
                json.dump(self.model_info_dict, fe, indent=4)

        elif self.is_test:
            df = pd.DataFrame(self.result_dict)
            df.to_csv(self.csv_save_dir + f'/evals_by_freq_test.csv')

            df = pd.DataFrame(self.sample_dict)
            df.to_csv(self.csv_save_dir + f'/evals_by_freq_sample_test.csv')            

    def dump_input_len(self):
        model_info_save_loc = self.model_info_save_dir  + '/model_info.json'
        with open(model_info_save_loc, 'w') as fe:
            json.dump(self.model_info_dict, fe, indent=4)