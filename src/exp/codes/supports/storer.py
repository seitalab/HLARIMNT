
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
            save_dir (str) : Path of the upper directory to save results.
            lowest_dir (str) : Path of the lower directory to save results.
            hla_list (List) : List of HLA genes.
        Returns:
            None
        """
        self.lowest_dir = lowest_dir
        self.hla_list = hla_list
        self.model_save_dir = save_dir + config['model_save_dir'] + lowest_dir
        self.log_save_dir = save_dir + config['log_save_dir'] + lowest_dir

        makedirs(self.model_save_dir)
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
            loss_dict (Dict): 
            acc_dict (Dict): 
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
        Args:
            modeldict (Dict): Models.
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
            is_test (bool) : 
        Returns:
            None
        """
        self.is_test = is_test
        self.csv_save_dir = save_dir + config['csv_save_dir']
        makedirs(self.csv_save_dir)
        self.model_info_save_dir = save_dir + config['model_save_dir']

        if is_test:
            self.result_dict = {'hla':[], 'digit':[], 'allele':[], 'r2':[], 'ppv':[], 'sensitivity':[], 'probability':[]}
        else:
            self.model_info_dict = {}

    def store_hla_list(self, idx, digit, hla_list):
        if digit == '2-digit':
            self.model_info_dict[idx]['hla_list'] = {}

        self.model_info_dict[idx]['hla_list'][digit] = hla_list

    def prepare_to_store_model_info(self, idx):
        self.model_info_dict[idx] = {}

    def store_input_len(self, idx, input_len, chunk_len):
        self.model_info_dict[idx]['input_len'] = input_len
        self.model_info_dict[idx]['chunk_len'] = chunk_len

    def store_evals(self, result_dict_tmp: Dict) -> None:
        """
        Store values of indices and frequency of each allele
        Args:
            result_dict_tmp (Dict) : 
        Returns:
            None
        """
        for key in result_dict_tmp.keys():
            self.result_dict[key] += result_dict_tmp[key]

    def dump_results(self) -> None:
        """
        Dump values of indices for each allele as csv file.
        Args:
            None
        Returns;
            None
        """
        if not self.is_test:
            model_info_save_loc = self.model_info_save_dir  + '/model_info.json'
            with open(model_info_save_loc, 'w') as fe:
                json.dump(self.model_info_dict, fe, indent=4)

        elif self.is_test:
            df = pd.DataFrame(self.result_dict)
            df.to_csv(self.csv_save_dir + f'/test_evals.csv')

    def dump_input_len(self):
        model_info_save_loc = self.model_info_save_dir  + '/model_info.json'
        with open(model_info_save_loc, 'w') as fe:
            json.dump(self.model_info_dict, fe, indent=4)
