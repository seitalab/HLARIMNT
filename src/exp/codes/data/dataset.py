
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from codes.supports.utils import *
import pickle
from typing import Dict, List, Tuple
from torch.utils.data.dataset import Subset
import random
import math
import json

config_file = '../config.yaml'

with open(config_file) as f:
    config = yaml.safe_load(f)['exp']

class DataProcessor:

    def __init__(self, params: Dict, logger: str) -> None:
        """
        Args:
            params : Dict of the setting of the experiment.
            logger : Path to the log file.
        Returns:
            None
        """
        self.params = params
        self.logger = logger
        self.digit_list = params['digits']

        ref = f"codes/data/{self.params['data_dir']}/{params['ref_name']}.bim"
        sample = f"codes/data/{self.params['data_dir']}/{params['ref_name']}_sample.bim"
        phased = f"codes/data/{self.params['data_dir']}/{params['ref_name']}.bgl.phased"

        self.ref_bim = pd.read_table(ref, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
        self.sample_bim = pd.read_table(sample, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
        ref_phased = pd.read_table(phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
        self.ref_phased = ref_phased.set_index(1)

        self.pc_data_loc = f"{params['ref_name']}" + config['pc_data_dir']
        makedirs(self.pc_data_loc)

        hla_info_loc = f'codes/data/{self.params["data_dir"]}/hla_info.json'

        with open(hla_info_loc) as f:
            self.hla_info = json.load(f)

    def _extract_concord(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        concord_snp = self.ref_bim.pos.isin(self.sample_bim.pos)
        model_bim_tmp = self.ref_bim.iloc[np.where(concord_snp)[0]]
        duplicated_num = len(model_bim_tmp[model_bim_tmp.duplicated(subset='pos')])
        
        self.num_concord = np.sum(concord_snp) - duplicated_num
        
        for i in range(self.num_concord):
            if concord_snp.iloc[i]:
                tmp = np.where(self.sample_bim.pos == self.ref_bim.iloc[i].pos)[0][0]
                if set((self.ref_bim.iloc[i].a1, self.ref_bim.iloc[i].a2)) != \
                        set((self.sample_bim.iloc[tmp].a1, self.sample_bim.iloc[tmp].a2)):
                    concord_snp.iloc[i] = False

        self.num_ref = self.ref_phased.shape[1] // 2
        self.ref_concord_phased = self.ref_phased.iloc[np.where(concord_snp)[0]]

        self.logger.log(f'{self.num_ref} individuals are loaded from the reference.')
        self.logger.log(f'{self.num_concord} SNPs are matched in position and used for training.')

        model_bim = self.ref_bim.iloc[np.where(concord_snp)[0]]
        model_bim = model_bim[~model_bim.duplicated(subset='pos')]
        model_bim.to_csv(self.pc_data_loc + f'/model.bim', sep='\t', header=False, index=False)

        self.model_bim = pd.read_table(self.pc_data_loc + f'/model.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')

    def _encode_snp(self) -> np.ndarray:
        """
        Args:
            None
        Returns:
            snp_encoded (np.ndarray)
        """
        snp_encoded = np.zeros((2*self.num_ref, self.num_concord, 2))
        for i in range(self.num_concord):
            a1 = self.model_bim.iloc[i].a1
            a2 = self.model_bim.iloc[i].a2
            snp_encoded[self.ref_concord_phased.iloc[i, :] == a1, i, 0] = 1 
            snp_encoded[self.ref_concord_phased.iloc[i, :] == a2, i, 1] = 1
        return snp_encoded
        
    def _encode_hla(self) -> np.ndarray:
        """
        Args:
            None 
        Returns:
            hla_encoded (np.ndarray)
        """
        hla_encoded = {}
        for hla in self.hla_info:
            for i in range(2*self.num_ref):
                hla_encoded[hla] = {}
            for digit in self.digit_list:
                hla_encoded[hla][digit] = np.zeros(2 * self.num_ref)
                for j in range(len(self.hla_info[hla][digit])):
                    hla_encoded[hla][digit][np.where(self.ref_phased.loc[self.hla_info[hla][digit][j]] == 'P')[0]] = j

        return hla_encoded

    def _dump_encoded_data(self) -> None:
        """
        Args: 
            None
        Returns:
            None 
        """
        self._extract_concord()
        self.snp_encoded = self._encode_snp()
        self.hla_encoded = self._encode_hla()

        with open(self.pc_data_loc + f'/encoded_snp.pkl', 'wb') as path:
            pickle.dump(self.snp_encoded, path)
        
        with open(self.pc_data_loc + '/encoded_hla.pkl', 'wb') as path:
            pickle.dump(self.hla_encoded, path)       

    def _set_index(self, hla_list: List) -> None:
        """
        Args: 
            hla_list (List) :
        Returns:
            st_index, ed_index (Tuple) : 
        """
        w = self.params['w']
        st = int(self.hla_info[hla_list[0]]['pos']) - w*1000
        ed = int(self.hla_info[hla_list[-1]]['pos']) + w*1000
        st_index = max(0, np.sum(self.model_bim.pos < st) - 1)
        ed_index = min(self.num_concord, self.num_concord - np.sum(self.model_bim.pos > ed))
        return st_index, ed_index

    def make_train_data(self, digit: str, hla_list: List) -> List:
        """
        Args:
            digit (str) :
            hla_list (List) : 
        Returns:
            train_data (List)
        """
        
        if os.path.exists(self.pc_data_loc + f'/encoded_snp.pkl'):
            with open(self.pc_data_loc + f'/encoded_snp.pkl', 'rb') as f:
                self.snp_encoded = pickle.load(f)

            with open(self.pc_data_loc + '/encoded_hla.pkl', 'rb') as f:
                self.hla_encoded = pickle.load(f)

            self.model_bim = pd.read_table(self.pc_data_loc + f'/model.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            self.num_concord = len(self.model_bim)
            self.num_ref = self.ref_phased.shape[1] // 2
        
        else:        
            self._dump_encoded_data()

        train_data = []
        st_index, ed_index = self._set_index(hla_list)
        data_num = self.snp_encoded.shape[0]
        for i in range(data_num):
            snp_encoded_tmp = self.snp_encoded[i, st_index:ed_index]
            chunk_spilt_num = self.params['chunk_split_num']
            chunk_len = math.ceil(snp_encoded_tmp.shape[0]/chunk_spilt_num)
            zero_snp_num =  chunk_spilt_num * chunk_len - snp_encoded_tmp.shape[0]
            snp_encoded_tmp = np.concatenate([snp_encoded_tmp, np.zeros((zero_snp_num, self.snp_encoded.shape[2]))], 0)
            snp_encoded_tmp = np.reshape(snp_encoded_tmp, (chunk_spilt_num, chunk_len, self.snp_encoded.shape[2]))
            tmp = [snp_encoded_tmp]
            tmp[0] = np.insert(tmp[0], 0, 0, axis=0)
            for hla in hla_list:
                tmp.append(self.hla_encoded[hla][digit][i])
            train_data.append(tmp)
        return train_data

    def calc_skip_hlas(self, digit, hla_list):
        skip_hlas = []
        allele_cnts = {}
        for hla in hla_list:
            allele_cnts[hla] = len(self.hla_info[hla][digit])
            if allele_cnts[hla] == 1:
                skip_hlas.append(hla)
        return skip_hlas, allele_cnts

def make_loaders(params: Dict, train_data: List, id: int, digit: str, seed: int) -> Tuple:
    """
    Args:
        params (Dict) : Settings.
        train_data (List) : 
        id (int) : Number of the model.
        digit (str) :
        seed (int) : Fold number.
    """
    val_split = params['val_split']
    fold_num = params['fold_num']
    batch_size = params['batch_size']
    num_ref = len(train_data) // 2
    pc_data_loc = f"{params['ref_name']}" + config['pc_data_dir']

    if fold_num != 1:
        idx = np.arange(0, len(train_data))
        test_index = idx[np.fmod(idx, fold_num) == seed]
        test_loader = torch.utils.data.DataLoader(Subset(train_data, test_index), batch_size=batch_size, shuffle=False)
        not_test_index = idx[np.fmod(idx, fold_num) != seed]                
        
        interval = len(not_test_index)//int(len(not_test_index)*val_split)
        val_index = not_test_index[::interval]
        val_index = np.delete(val_index, [0])
        train_index = np.setdiff1d(not_test_index, val_index)

        val_loader = torch.utils.data.DataLoader(Subset(train_data, val_index), batch_size=batch_size, shuffle=False)
        train_loader = torch.utils.data.DataLoader(Subset(train_data, train_index), batch_size=batch_size)

    else:
        train_index = np.arange(int(2*num_ref*val_split), 2*num_ref)
        val_index = np.arange(int(2*num_ref*val_split))

        train_loader = torch.utils.data.DataLoader(Subset(train_data, train_index), batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(Subset(train_data, val_index), batch_size=batch_size, shuffle=False)
        test_loader = None

    loader_save_loc = pc_data_loc + f'/seed{seed}/id{id}-{digit}'
    makedirs(loader_save_loc)

    with open(loader_save_loc + '/train_loader.pkl', 'wb') as path:
        pickle.dump(train_loader, path)

    with open(loader_save_loc + '/val_loader.pkl', 'wb') as path:
        pickle.dump(val_loader, path)

    if fold_num != 1:
        with open(loader_save_loc + '/test_loader.pkl', 'wb') as path:
            pickle.dump(test_loader, path)

    return train_loader, val_loader, test_loader