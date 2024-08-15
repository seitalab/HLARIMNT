import numpy as np
import torch
import torch.nn as nn


from calendar import c
import yaml
from torch.utils.data import DataLoader, Dataset
from codes.supports.utils import *
import pickle
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data.dataset import Subset
import random
from codes.data.dataconvert import DataAugumentor
from sklearn.cluster import KMeans
import math

config_file = '../config.yaml'

with open(config_file) as f:
    config = yaml.safe_load(f)

class DataProcessor:

    def __init__(self, params: Dict, logger) -> None:
        """
        Args:
            args (Namespace) : Dict of the setting of the experiment.
        Returns:
            None
        """
        self.params = params
        self.logger = logger
        data_root = config['dataset']['save_root']
        dataset_name = params['data']['dataset']
        data_root = data_root + config['dataset'][dataset_name]['dirname']
        
        hla_info = data_root + config['dataset'][dataset_name]['hla_info']
        self.hla_info = path_to_dict(hla_info)
        pc_data_root = config['exp']['pc_data_root']
        self.pc_data_loc = pc_data_root + f'/{params["exp_name"]}'+ f'/{params["task"]}-{params["data"]["dataset"]}'
        makedirs(self.pc_data_loc)
        self.digit_list = params['digits']
        self.kmeans = params['kmeans']

        if self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC':
            pan_asian_ref = data_root + '/Mixed_Pan-Asian.bim'
            t1dgc_ref = data_root + '/Mixed_T1DGC.bim'
            pa_bim_for_bgl = data_root + '/Pan-Asian_REF_proc.bim'
            t1_bim_for_bgl = data_root + '/T1DGC_REF_proc2.bim'
            self.pan_asian_ref_bim = pd.read_table(pan_asian_ref, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            self.pa_bim_for_bgl = pd.read_table(pa_bim_for_bgl, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            self.t1dgc_ref_bim = pd.read_table(t1dgc_ref, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            self.t1_bim_for_bgl = pd.read_table(t1_bim_for_bgl, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')

            pan_asian_phased = data_root + '/Pan-Asian_REF.bgl.phased'
            t1dgc_phased = data_root + '/T1DGC_REF_proc2.bgl.phased'
            pan_asian_ref_phased = pd.read_table(pan_asian_phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
            if self.params['data']['dataset'] == 'Equal':
                t1dgc_ref_phased = pd.read_table(t1dgc_phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:1062]
            else:
                t1dgc_ref_phased = pd.read_table(t1dgc_phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
            self.pan_asian_phased = pan_asian_ref_phased.set_index(1)
            self.t1dgc_phased = t1dgc_ref_phased.set_index(1)

        else:
            ref = data_root + params['data']['ref']
            #sample = data_root + params['data']['sample']
            self.ref_bim = pd.read_table(ref, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            phased = data_root + params['data']['phased']
            
            if self.params['data']['dataset'] == 'T1DGC_530':
                ref_phased = pd.read_table(phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:1062]
            if self.params['data']['dataset'] == 'T1DGC_1300':
                ref_phased = pd.read_table(phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:2602]
            if self.params['data']['dataset'] == 'T1DGC_2600':
                ref_phased = pd.read_table(phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:5202]
            else:
                ref_phased = pd.read_table(phased, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
            
            self.ref_phased = ref_phased.set_index(1)
            #self.sample_bim = pd.read_table(sample, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            #assert self.kmeans == False

    def _make_sample_bim(self, collapse_rate):
        if self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC':
            sample_bim_tmp_pa = self.pan_asian_ref_bim[~(self.pan_asian_ref_bim['id'].str.startswith('SNP')) & ~(self.pan_asian_ref_bim['id'].str.startswith('AA')) & ~(self.pan_asian_ref_bim['id'].str.startswith('HLA')) & ~(self.pan_asian_ref_bim['id'].str.startswith('INS'))]
            sample_bim_tmp_t1 = self.t1dgc_ref_bim[~(self.t1dgc_ref_bim['id'].str.startswith('SNP')) & ~(self.t1dgc_ref_bim['id'].str.startswith('AA')) & ~(self.t1dgc_ref_bim['id'].str.startswith('HLA')) & ~(self.t1dgc_ref_bim['id'].str.startswith('INS'))]
        #collapsed = [(collapse_rate <= random.random()) for _ in range(len(sample_bim_tmp))]

            return sample_bim_tmp_pa.sample(int(len(sample_bim_tmp_pa)*(1.0 - collapse_rate))), sample_bim_tmp_t1.sample(int(len(sample_bim_tmp_t1)*(1.0 - collapse_rate)))
        #sample_bim_tmp = self.ref_bim[self.ref_bim['id'].str.startwith('rs')]
        else:
            sample_bim_tmp = self.ref_bim[~(self.ref_bim['id'].str.startswith('SNP')) & ~(self.ref_bim['id'].str.startswith('AA')) & ~(self.ref_bim['id'].str.startswith('HLA')) & ~(self.ref_bim['id'].str.startswith('INS'))]
        #collapsed = [(collapse_rate <= random.random()) for _ in range(len(sample_bim_tmp))]

            return sample_bim_tmp.sample(int(len(sample_bim_tmp)*(1.0 - collapse_rate)))#[collapsed]

    def _extract_concord(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        collapse_rate = self.params['collapse'][0]
        if self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC':
            self.sample_bim, self.t1_sample_bim = self._make_sample_bim(collapse_rate)

            concord_snp_pa = self.pan_asian_ref_bim.pos.isin(self.sample_bim.pos)
            concord_snp_t1 = self.t1dgc_ref_bim.pos.isin(self.t1_sample_bim.pos)

            concord_snp_pa_bgl = self.pa_bim_for_bgl.pos.isin(self.sample_bim.pos)
            concord_snp_t1_bgl = self.t1_bim_for_bgl.pos.isin(self.t1_sample_bim.pos)

            self.num_concord = np.sum(concord_snp_pa)
            #pd.DataFrame(concord_snp_pa).to_csv('concord_snp_pa.csv')
            #pd.DataFrame(concord_snp_t1).to_csv('concord_snp_t1.csv')

        else:
            self.sample_bim = self._make_sample_bim(collapse_rate)
            concord_snp = self.ref_bim.pos.isin(self.sample_bim.pos)
            self.num_concord = np.sum(concord_snp)
        #removed_bim = self.ref_bim[(self.ref_bim['id'].str.startswith('SNP')) | (self.ref_bim['id'].str.startswith('AA')) | (self.ref_bim['id'].str.startswith('HLA')) | (self.ref_bim['id'].str.startswith('INS'))]
        
            for i in range(self.num_concord):
                if concord_snp.iloc[i]:
                    tmp = np.where(self.sample_bim.pos == self.ref_bim.iloc[i].pos)[0][0]
                    if set((self.ref_bim.iloc[i].a1, self.ref_bim.iloc[i].a2)) != \
                            set((self.sample_bim.iloc[tmp].a1, self.sample_bim.iloc[tmp].a2)):
                        concord_snp.iloc[i] = False

        if not (self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC'):
            self.num_ref = self.ref_phased.shape[1] // 2
            self.ref_concord_phased = self.ref_phased.iloc[np.where(concord_snp)[0]]
        else:
            pan_asian_num_ref = self.pan_asian_phased.shape[1] // 2
            t1dgc_num_ref = self.t1dgc_phased.shape[1] // 2
            self.num_ref = pan_asian_num_ref + t1dgc_num_ref
            self.ref_concord_phased_pan_asian = self.pan_asian_phased.iloc[np.where(concord_snp_pa_bgl)[0]]
            self.ref_concord_phased_t1dgc = self.t1dgc_phased.iloc[np.where(concord_snp_t1_bgl)[0]]
            pd.DataFrame(self.ref_concord_phased_t1dgc).to_csv('t1dgc_phased.csv')
            pd.DataFrame(self.ref_concord_phased_pan_asian).to_csv('pa_phased.csv')

        self.logger.log(f'{self.num_ref} people loaded from reference.')
        if not (self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC'):
            self.logger.log(f'{len(self.ref_bim)} SNPs loaded from reference.')
        self.logger.log(f'{len(self.sample_bim)} SNPs loaded from sample.')
        self.logger.log(f'{self.num_concord} SNPs matched in position and used for training.')

        if not (self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC'):
            if self.params['encode'] == 'chunk' and self.params['chunk_all']:
                model_bim = self.ref_bim.iloc[np.where(concord_snp)[0]]
                pos_ids = self._add_cls_info(model_bim)
                model_bim['pos_id'] = pos_ids
                idx_ids = self._add_idx_info(model_bim)
                model_bim['idx_id'] = idx_ids
            else:
                model_bim = self.ref_bim.iloc[np.where(concord_snp)[0]]
            model_bim.to_csv(self.pc_data_loc + f'/model_{self.params["w"]}w_{self.kmeans}.bim', sep='\t', header=False, index=False)
            if self.params['encode'] == 'chunk' and self.params['chunk_all']:
                self.model_bim = pd.read_table(self.pc_data_loc + f'/model_{self.params["w"]}w_{self.kmeans}.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2', 'pos_id', 'idx_id'], header=None, engine='python')
            else:
                self.model_bim = pd.read_table(self.pc_data_loc + f'/model_{self.params["w"]}w_{self.kmeans}.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')

        else:
            pa_model_bim = self.pan_asian_ref_bim.iloc[np.where(concord_snp_pa)[0]]
            t1_model_bim = self.t1dgc_ref_bim.iloc[np.where(concord_snp_t1)[0]]
            pa_model_bim.to_csv(self.pc_data_loc + '/Pan-Asian_model.bim', sep='\t', header=False, index=False)
            t1_model_bim.to_csv(self.pc_data_loc + '/T1DGC_model.bim', sep='\t', header=False, index=False)
            self.pa_model_bim = pd.read_table(self.pc_data_loc + '/Pan-Asian_model.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            self.t1_model_bim = pd.read_table(self.pc_data_loc + '/T1DGC_model.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
            self.model_bim = pd.read_table(self.pc_data_loc + '/Pan-Asian_model.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')

    def _add_cls_info(self, model_bim):
        positions = np.array(model_bim['pos'].tolist(), np.int32)
        positions = positions.T
        if self.params['kmeans']:
            pos_ids = KMeans(n_clusters=self.params['w'], random_state=0).fit_predict(positions.reshape(-1, 1))
        else:
            """
                chunk_len = math.ceil(snp_encoded_tmp.shape[0]/chunk_spilt_num)
                zero_snp_num =  chunk_spilt_num * chunk_len - snp_encoded_tmp.shape[0]
                snp_encoded_tmp = np.concatenate([snp_encoded_tmp, np.zeros((zero_snp_num, self.snp_encoded.shape[2]))], 0)
                snp_encoded_tmp = np.reshape(snp_encoded_tmp, (chunk_spilt_num, chunk_len, self.snp_encoded.shape[2]))
                """
            chunk_len = math.ceil(positions.shape[0] / self.params['chunk_split_num'])
            print(chunk_len)
            pos_ids = np.repeat([i for i in range(self.params['chunk_split_num'])], chunk_len)
            print(len(pos_ids))
            pos_ids = pos_ids[:positions.shape[0]]
            #pos_ids = np.concatenate([pos_ids, np.array([(self.params['chunk_split_num']-1) for i in range(positions.shape[0]-pos_ids.shape[0])])],0)
        return pos_ids
        
    def _add_idx_info(self, model_bim):
        chunk_len = model_bim['pos_id'].value_counts().iloc[0]
        unique_ids = model_bim['pos_id'].unique().tolist()
        idx_ids = []
        for pos_id in unique_ids:
            snp_num = len(model_bim[model_bim['pos_id']==pos_id])
            all_ids = [i for i in range(chunk_len)]
            idx_ids_tmp = sorted(random.sample(all_ids, snp_num))
            idx_ids += idx_ids_tmp
        return idx_ids

    def _encode_snp(self) -> Dict:
        """
        Args:
            None
        Returns:
            snp_encoded (Dict)
        """
        self.logger.log(f'SNPs are newly encoded by type {self.params["encode"]}.')

        if self.params['encode'] == 'deep_hla' or self.params['encode'] == 'input_conv':
            snp_encoded = np.zeros((2*self.num_ref, self.num_concord, 2))
            for i in range(self.num_concord):
                a1 = self.model_bim.iloc[i].a1
                a2 = self.model_bim.iloc[i].a2
                snp_encoded[self.ref_concord_phased.iloc[i, :] == a1, i, 0] = 1 
                snp_encoded[self.ref_concord_phased.iloc[i, :] == a2, i, 1] = 1
            return snp_encoded

        if self.params['encode'] == 'chunk':
            snp_encoded_tmp = np.zeros((2*self.num_ref, self.num_concord, 2))
            for i in range(self.num_concord):
                a1 = self.model_bim.iloc[i].a1
                a2 = self.model_bim.iloc[i].a2
                snp_encoded_tmp[self.ref_concord_phased.iloc[i, :] == a1, i, 0] = 1 
                snp_encoded_tmp[self.ref_concord_phased.iloc[i, :] == a2, i, 1] = 1

            if not self.params['chunk_all']:
                return snp_encoded_tmp
            
            else:
                chunk_len = self.model_bim['pos_id'].value_counts().iloc[0]
                pos_ids = self.model_bim['pos_id'].unique().tolist()

                for i, pos_id in enumerate(pos_ids):
                    indice_of_snp = self.model_bim[self.model_bim['pos_id']==pos_id].index.tolist()
                    st = indice_of_snp[0]
                    ed = indice_of_snp[-1]
                    snps = snp_encoded_tmp[:, st:ed+1, :]
                    chunk = np.zeros((snps.shape[0], chunk_len, snps.shape[2]))
                    snp_loc = np.array(self.model_bim[st:ed+1]['idx_id'].tolist())
                    chunk[:, snp_loc, :] = snps
                    if i==0:
                        snp_encoded = chunk.copy()
                        snp_encoded = np.reshape(snp_encoded, (snp_encoded.shape[0], 1, snp_encoded.shape[1], snp_encoded.shape[2]))
                        #print(snp_encoded.shape)
                    else:
                        snp_encoded = np.concatenate([snp_encoded, np.reshape(chunk, (chunk.shape[0],1,chunk.shape[1], chunk.shape[2]))], axis=1)
                    #print(snp_encoded.shape)
                zero_chunk_num = self.params['chunk_split_num'] - len(pos_ids)
                for i in range(zero_chunk_num):
                    chunk = np.zeros((snp_encoded_tmp.shape[0], 1, chunk_len, snp_encoded_tmp.shape[2]))
                    snp_encoded = np.concatenate([snp_encoded, chunk], axis=1)
                return snp_encoded
        
        elif self.params['encode'] == 'by_input_len':
            snp_encoded = np.zeros((2*self.num_ref, self.num_concord))
            for i in range(self.num_concord):
                a1 = self.model_bim.iloc[i].a1
                a2 = self.model_bim.iloc[i].a2
                snp_encoded[self.ref_concord_phased.iloc[i, :] == a1, i] = 2 * i + 2
                snp_encoded[self.ref_concord_phased.iloc[i, :] == a2, i] = 2 * i + 3
            return snp_encoded

        elif self.params['encode'] == 'by_input_len_base':
            snp_encoded = np.zeros((2*self.num_ref, self.num_concord))
            for i in range(self.num_concord):
                a1 = self.model_bim.iloc[i].a1
                a2 = self.model_bim.iloc[i].a2
                if a1 not in ['A', 'G', 'C', 'T'] or a2 not in ['A', 'G', 'C', 'T']:
                    print(a1, a2, i)

                for id, base in enumerate(['A', 'G', 'C', 'T']):
                    if base == a1:
                        snp_encoded[self.ref_concord_phased.iloc[i, :] == base, i] = 8 * i + 2 * id + 2
                    elif base == a2:
                        snp_encoded[self.ref_concord_phased.iloc[i, :] == base, i] = 8 * i + 2 * id + 3
            return snp_encoded

        elif self.params['encode'] == '2dim':
            snp_encoded = np.zeros((2*self.num_ref, self.num_concord))
            for i in range(self.num_concord):
                a1 = self.model_bim.iloc[i].a1
                a2 = self.model_bim.iloc[i].a2
                snp_encoded[self.ref_concord_phased.iloc[i, :] == a1, i] = 2
                snp_encoded[self.ref_concord_phased.iloc[i, :] == a2, i] = 3
            return snp_encoded
        
        elif self.params['encode'] == '4dim':
            snp_encoded = np.zeros((2*self.num_ref, self.num_concord))
            for i in range(self.num_concord):
                snp_encoded[self.ref_concord_phased.iloc[i, :] == 'A', i] = 2
                snp_encoded[self.ref_concord_phased.iloc[i, :] == 'G', i] = 3
                snp_encoded[self.ref_concord_phased.iloc[i, :] == 'C', i] = 4
                snp_encoded[self.ref_concord_phased.iloc[i, :] == 'T', i] = 5  
            return snp_encoded

        elif self.params['encode'] == '8dim':
            snp_encoded = np.zeros((2*self.num_ref, self.num_concord))
            for i in range(self.num_concord):
                a1 = self.model_bim.iloc[i].a1
                a2 = self.model_bim.iloc[i].a2
                id = 0
                for base in ['A', 'G', 'C', 'T']:
                    if base == a1:
                        snp_encoded[self.ref_concord_phased.iloc[i, :] == base, i] = id * 2 + 2
                    elif base == a2:
                        snp_encoded[self.ref_concord_phased.iloc[i, :] == base, i] = id * 2 + 3
                    id += 1
            return snp_encoded

    def _encode_hla(self) -> Dict:
        """
        Args:
            None 
        Returns:
            hla_encoded (Dict)
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

    def _encode_snp_mixed(self, data_name):
        #assert self.params['encode'] == 'chunk'
        if data_name == 'Pan-Asian':
            num_ref = self.pan_asian_phased.shape[1] // 2
            refcon = self.ref_concord_phased_pan_asian
            model_bim = self.pa_model_bim
        elif data_name == 'T1DGC':
            num_ref = self.t1dgc_phased.shape[1] // 2
            refcon = self.ref_concord_phased_t1dgc
            model_bim = self.t1_model_bim
        snp_encoded_tmp = np.zeros((2*num_ref, self.num_concord, 2))
        for i in range(self.num_concord):
            a1 = model_bim.iloc[i].a1
            a2 = model_bim.iloc[i].a2
            snp_encoded_tmp[refcon.iloc[i, :] == a1, i, 0] = 1 
            snp_encoded_tmp[refcon.iloc[i, :] == a2, i, 1] = 1
        return snp_encoded_tmp

    def _encode_hla_mixed(self, data_name) -> Dict:
        """
        Args:
            None 
        Returns:
            hla_encoded (Dict)
        """
        if data_name == 'Pan-Asian':
            num_ref = self.pan_asian_phased.shape[1] // 2
            ref_phased = self.pan_asian_phased
            pa_hla_info_path = '/root/export/users/kaho/prime_hla/data/Pan-Asian/Pan-Asian_hla_info.yaml'
            with open(pa_hla_info_path) as f:
                ds_hla_info = yaml.safe_load(f)
        elif data_name == 'T1DGC':
            num_ref = self.t1dgc_phased.shape[1] // 2
            ref_phased = self.t1dgc_phased
            t1_hla_info_path = '/root/export/users/kaho/prime_hla/data/T1DGC/T1DGC_hla_info.yaml'
            with open(t1_hla_info_path) as f:
                ds_hla_info = yaml.safe_load(f)

        hla_encoded = {}
        for hla in self.hla_info:
            for i in range(2*num_ref):
                hla_encoded[hla] = {}
            for digit in self.digit_list:
                if self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC':
                    hla_encoded[hla][digit] = np.ones(2 * num_ref)*(-1)
                else:
                    hla_encoded[hla][digit] = np.zeros(2 * num_ref)
                for j in range(len(self.hla_info[hla][digit])):
                    if self.hla_info[hla][digit][j] in ds_hla_info[hla][digit]:
                        hla_encoded[hla][digit][np.where(ref_phased.loc[self.hla_info[hla][digit][j]] == 'P')[0]] = j

        return hla_encoded


    def _dump_encoded_data(self) -> None:
        """
        Args: 
            None
        Returns:
            None 
        """
        self._extract_concord()
        snp_encode_type = self.params["encode"]
        if self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC':
            self.pan_asian_snp_encoded = self._encode_snp_mixed('Pan-Asian')
            self.t1dgc_snp_encoded = self._encode_snp_mixed('T1DGC')
            self.pan_asian_hla_encoded = self._encode_hla_mixed('Pan-Asian')
            self.t1dgc_hla_encoded = self._encode_hla_mixed('T1DGC')
            with open(self.pc_data_loc + f'/encoded_snp_pan_asian.pkl', 'wb') as path:
                pickle.dump(self.pan_asian_snp_encoded, path)
            with open(self.pc_data_loc + '/encoded_hla_pan_asian.pkl', 'wb') as path:
                pickle.dump(self.pan_asian_hla_encoded, path)
            with open(self.pc_data_loc + f'/encoded_snp_t1dgc.pkl', 'wb') as path:
                pickle.dump(self.t1dgc_snp_encoded, path)
            with open(self.pc_data_loc + '/encoded_hla_t1dgc.pkl', 'wb') as path:
                pickle.dump(self.t1dgc_hla_encoded, path)
        else:
            self.snp_encoded = self._encode_snp()
            self.hla_encoded = self._encode_hla()

            with open(self.pc_data_loc + f'/encoded_snp_{self.params["w"]}w_{self.kmeans}.pkl', 'wb') as path:
                pickle.dump(self.snp_encoded, path)
            
            with open(self.pc_data_loc + '/encoded_hla.pkl', 'wb') as path:
                pickle.dump(self.hla_encoded, path)

    def _set_index(self, hla_list: List) -> None:
        """
        Args: 
            hla_list (List) :
        Returns:
            indice (Tuple) : 
        """
        #w = self.params['w'] * 1000
        
        if self.params['encode'] == 'chunk' and self.params['chunk_all']:
            st_index = 0
            ed_index = self.params['chunk_split_num']
        else:
            w = self.params['w']
            st = int(self.hla_info[hla_list[0]]['pos']) - w*1000
            ed = int(self.hla_info[hla_list[-1]]['pos']) + w*1000
            st_index = max(0, np.sum(self.model_bim.pos < st) - 1)
            ed_index = min(self.num_concord, self.num_concord - np.sum(self.model_bim.pos > ed))
            #ed_index = self.num_concord
        """
        else:
            w_half = self.params['w'] // 2

            gene_pos = int(self.hla_info[hla_list[0]]['pos'])

            #model_bim_tmp = self.model_bim.reset_index()
            #gene_idx = self.model_bim[self.model_bim.pos==gene_pos].index
            nearest = self.model_bim.iloc[(self.model_bim['pos']-gene_pos).abs().argsort()[:2]]
            middle_pos = nearest['pos'].tolist()[0]
            middle_idx = nearest.index.tolist()[0]

            st_index = max(0, middle_idx-w_half)
            ed_index = min(self.num_concord, middle_idx+w_half)
            """
            
        """
        if self.params['task'] == 'cnn_prev' or self.params['task'] == 'cnn_best' or (self.params['encode']=='input_conv' and self.params['chunk_all']):
            st_index = 0
            ed_index = self.num_concord
            """

        return st_index, ed_index

    def make_train_data(self, digit: str, hla_list: List) -> List:
        """
        data augumentation, kernel PCA etc. are executed here.
        Args:
            digit (str) :
            hla_list (List) : 
        Returns:
            train_data (List)
        """
        snp_encode_type = self.params["encode"]

        if self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal' or self.params['data']['dataset'] == 'Ind_Pan-Asian' or self.params['data']['dataset'] == 'Ind_T1DGC':
            if os.path.exists(self.pc_data_loc + f'/encoded_snp_pan_asian.pkl'):
                with open(self.pc_data_loc + f'/encoded_snp_pan_asian.pkl', 'rb') as f:
                    self.pan_asian_snp_encoded = pickle.load(f)
                with open(self.pc_data_loc + f'/encoded_snp_t1dgc.pkl', 'rb') as f:
                    self.t1dgc_snp_encoded = pickle.load(f)
                with open(self.pc_data_loc + '/encoded_hla_pan_asian.pkl', 'rb') as f:
                    self.pan_asian_hla_encoded = pickle.load(f)  
                with open(self.pc_data_loc + '/encoded_hla_t1dgc.pkl', 'rb') as f:
                    self.t1dgc_hla_encoded = pickle.load(f)    
                self.model_bim = pd.read_table(self.pc_data_loc + f'/Pan-Asian_model.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
                self.num_concord = len(self.model_bim)
                pan_asian_num_ref = self.pan_asian_phased.shape[1] // 2
                t1dgc_num_ref = self.t1dgc_phased.shape[1] // 2
                self.num_ref = pan_asian_num_ref + t1dgc_num_ref
            else:
                self._dump_encoded_data()
        else:
            if os.path.exists(self.pc_data_loc + f'/encoded_snp_{self.params["w"]}w_{self.kmeans}.pkl'):
                with open(self.pc_data_loc + f'/encoded_snp_{self.params["w"]}w_{self.kmeans}.pkl', 'rb') as f:
                    self.snp_encoded = pickle.load(f)

                with open(self.pc_data_loc + '/encoded_hla.pkl', 'rb') as f:
                    self.hla_encoded = pickle.load(f)
                if self.params['encode'] == 'chunk' and self.params['chunk_all']:
                    self.model_bim = pd.read_table(self.pc_data_loc + f'/model_{self.params["w"]}w_{self.kmeans}.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2', 'pos_id', 'idx_id'], header=None, engine='python')
                else:
                    self.model_bim = pd.read_table(self.pc_data_loc + f'/model_{self.params["w"]}w_{self.kmeans}.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
                self.num_concord = len(self.model_bim)
                self.num_ref = self.ref_phased.shape[1] // 2
                print(self.snp_encoded)
            
            else:        
                self._dump_encoded_data()

        train_data = []
        st_index, ed_index = self._set_index(hla_list)
        self.logger.log(f'{hla_list}:st={st_index},ed={ed_index}')
        """
        if self.params['encode'] == 'chunk' and  self.params['chunk_all']:
            chunk_len = self.model_bim['pos_id'].value_counts().iloc[0]
        elif self.params['encode'] == 'chunk' and  not self.params['chunk_all']:
            snp_encoded_tmp = self.snp_encoded[i, st_index:ed_index]
            chunk_spilt_num = self.params['chunk_split_num']
            chunk_len = math.ceil((ed_index - st_index)/chunk_spilt_num)
            """
        if self.params['data']['dataset'] == 'Mixed' or self.params['data']['dataset'] == 'Equal':
            self.snp_encoded = np.concatenate([self.pan_asian_snp_encoded, self.t1dgc_snp_encoded],0)
            self.hla_encoded = {}
            for hla in hla_list:
                self.hla_encoded[hla] = {}
                self.hla_encoded[hla][digit] = np.array(list(self.pan_asian_hla_encoded[hla][digit]) + list(self.t1dgc_hla_encoded[hla][digit]))

        elif self.params['data']['dataset'] == 'Ind_Pan-Asian':
            self.snp_encoded = self.pan_asian_snp_encoded
            self.hla_encoded = {}
            for hla in hla_list:
                self.hla_encoded[hla] = {}
                self.hla_encoded[hla][digit] = self.pan_asian_hla_encoded[hla][digit]

        elif self.params['data']['dataset'] == 'Ind_T1DGC':
            self.snp_encoded = self.t1dgc_snp_encoded
            self.hla_encoded = {}
            for hla in hla_list:
                self.hla_encoded[hla] = {}
                self.hla_encoded[hla][digit] = self.t1dgc_hla_encoded[hla][digit]

        #######self.snp_encoded = np.cancatenate(pan_asian_snp_encoded, t1dgc_snp_encoded)
        if self.params['exp_name'] == 'exp10_T1DGC':
            data_num = 2*self.num_ref // 10
        else:
            data_num = self.snp_encoded.shape[0]
        for i in range(data_num):
            if self.params['encode'] == 'by_input_len':
                tmp = [self.snp_encoded[i, st_index:ed_index] - st_index * 2]
            elif self.params['encode'] == 'by_input_len_base':
                tmp = [self.snp_encoded[i, st_index:ed_index] - st_index * 8]
            elif self.params['encode'] == 'chunk' and not self.params['chunk_all']:
                snp_encoded_tmp = self.snp_encoded[i, st_index:ed_index]
                chunk_spilt_num = self.params['chunk_split_num']
                """
                if self.kmeans:
                    positions = np.array(self.model_bim['pos'].tolist()[st_index, ed_index], np.int32)
                    positions = positions.T
                    pos_ids = KMeans(n_clusters=chunk_spilt_num, random_state=0).fit_predict(positions.reshape(-1, 1))
                    pos_id_unique = list(dict.fromkeys(pos_ids))
                    for pos_id in pos_id_unique:
                        """

                chunk_len = math.ceil(snp_encoded_tmp.shape[0]/chunk_spilt_num)
                zero_snp_num =  chunk_spilt_num * chunk_len - snp_encoded_tmp.shape[0]
                snp_encoded_tmp = np.concatenate([snp_encoded_tmp, np.zeros((zero_snp_num, self.snp_encoded.shape[2]))], 0)
                snp_encoded_tmp = np.reshape(snp_encoded_tmp, (chunk_spilt_num, chunk_len, self.snp_encoded.shape[2]))
                tmp = [snp_encoded_tmp]
                
            else:
                #print(self.snp_encoded.shape)
                tmp = [self.snp_encoded[i, st_index:ed_index]]

            if not self.params['encode'] == 'deep_hla' and not self.params['encode'] == 'chunk' and not self.params['encode']=='input_conv':
                tmp[0] = np.insert(tmp[0], 0, 1)
            if self.params['encode'] == 'chunk':
                #print(np.ones((tmp[0][0].shape[0], tmp[0][0].shape[1])).shape)
                tmp[0] = np.insert(tmp[0], 0, 0, axis=0)
                #print(tmp[0].shape)
                #a = tmp[0][0].copy()
                #print(a.reshape(1, tmp[0][0].shape[0], tmp[0][0].shape[1]).shape)
                #print(np.array(np.random.random_sample(a.reshape(1, a.shape[0], a.shape[1]).shape)).shape)
                #tmp[0] = np.insert(tmp[0], 0, cls_token, axis=0)
                #print(tmp[0].shape)
            add_as_data = True
            for hla in hla_list:
                tmp.append(self.hla_encoded[hla][digit][i])
                if self.hla_encoded[hla][digit][i] == -1:
                    add_as_data = False
            if add_as_data:
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

class DataSplitter:
    """
    一つの遺伝子について、訓練、テストデータにそれぞれ全種類の配列が含まれるようにする
    """
    def __init__(self, params, id, seed) -> None:
        self.params = params
        self.hla = params['model']['grouping'][id][0]
        self.seed = seed
        data_root = config['dataset']['save_root']
        dataset_name = params['data']['dataset']
        data_root = data_root + config['dataset'][dataset_name]['dirname']
        
        hla_info = data_root + config['dataset'][dataset_name]['hla_info']
        self.hla_info = path_to_dict(hla_info)

    def _classify_data_by_allele(self, hla_encoded):
        """
        4-digitアレルの種類ごとにデータを割り振る
        {0:ndarray[1,3,6], 'allele2':[...]}
        """
        max_digit = self.params['digits'][-1]
        
        allele_ids = hla_encoded[self.hla][max_digit]
        max_id = len(self.hla_info[self.hla][max_digit])

        classified_data = {}
        for allele_id in range(max_id):
            classified_data[allele_id] = np.where(allele_ids==allele_id)[0]

        return classified_data

    def calc_test_idx(self, hla_encoded):
        classified_data = self._classify_data_by_allele(hla_encoded)

        test_idx = []
        fold_num = self.params['fold_num']
        for allele_id in classified_data:
            allele_ids = classified_data[allele_id]
            if allele_ids.shape[0] <= 1:
                continue

            elif allele_ids.shape[0] < fold_num:
                if self.seed < allele_ids.shape[0]:
                    test_idx.append(allele_ids[self.seed])
                else:
                    test_idx.append(allele_ids[random.randint(0, allele_ids.shape[0]-1)])

            else:
                idx = np.arange(allele_ids.shape[0])
                test_idx += list(allele_ids[np.fmod(idx, fold_num) == self.seed])

        return np.array(test_idx)

def make_loaders(params: Dict, train_data: List, id: int, digit: str, seed: int) -> Tuple:
    val_split = params['val_split']
    fold_num = params['fold_num']
    batch_size = params['batch_size']
    num_ref = len(train_data) // 2

    if fold_num != -1:
        idx = np.arange(0, len(train_data))
        if params['use_splitter']:
            splitter = DataSplitter(params, id, seed)
            pc_data_root_tmp = config['exp']['pc_data_root']
            pc_data_loc = pc_data_root_tmp + f'/{params["exp_name"]}'+ f'/{params["task"]}-{params["data"]["dataset"]}'
            with open(pc_data_loc + '/encoded_hla.pkl', 'rb') as f:
                hla_encoded = pickle.load(f)
            test_index = splitter.calc_test_idx(hla_encoded)
            not_test_index = np.setdiff1d(idx, test_index)
            test_loader = torch.utils.data.DataLoader(Subset(train_data, test_index), batch_size=batch_size, shuffle=False)
        else:
            not_test_index = idx[np.fmod(idx, fold_num) != seed]
            if not params['exp_name'] == 'test_loader':
                test_index = idx[np.fmod(idx, fold_num) == seed]
            else:
                test_index = np.arange(2*num_ref)
            test_loader = torch.utils.data.DataLoader(Subset(train_data, test_index), batch_size=batch_size, shuffle=False)                
        pc_data_root = config['exp']['pc_data_root']
        pc_data_loc = pc_data_root + f'/{params["exp_name"]}'
        if params['exp_name'] == 'T1DGC_530':
            if os.path.exists(pc_data_loc + f'/training_indice_{seed}.pkl'):
                with open(pc_data_loc + f'/training_indice_{seed}.pkl', 'rb') as f:
                    not_test_index = pickle.load(f)
            else:
                not_test_index = np.sort(np.random.choice(not_test_index, size=(1060,),replace=False))
                with open(pc_data_loc + f'/training_indice_{seed}.pkl', 'wb') as path:
                    pickle.dump(not_test_index, path)

        elif params['exp_name'] == 'T1DGC_1300':
            if os.path.exists(pc_data_loc + f'/training_indice_{seed}.pkl'):
                with open(pc_data_loc + f'/training_indice_{seed}.pkl', 'rb') as f:
                    not_test_index = pickle.load(f)
            else:
                not_test_index = np.sort(np.random.choice(not_test_index, size=(2600,),replace=False))
                with open(pc_data_loc + f'/training_indice_{seed}.pkl', 'wb') as path:
                    pickle.dump(not_test_index, path)

        elif params['exp_name'] == 'T1DGC_2600':
            if os.path.exists(pc_data_loc + f'/training_indice_{seed}.pkl'):
                with open(pc_data_loc + f'/training_indice_{seed}.pkl', 'rb') as f:
                    not_test_index = pickle.load(f)
            else:
                not_test_index = np.sort(np.random.choice(not_test_index, size=(5200,),replace=False))
                with open(pc_data_loc + f'/training_indice_{seed}.pkl', 'wb') as path:
                    pickle.dump(not_test_index, path)
        
        not_test_data = Subset(train_data, not_test_index)
        if params['data_aug']:
            augumentor = DataAugumentor(params, id, digit)
            not_test_data_auged = augumentor.augument(not_test_data)

            train_index = np.arange(int(len(not_test_data_auged)*val_split), len(not_test_data_auged))
            val_index = np.arange(int(len(not_test_data_auged)*val_split))

            val_loader = torch.utils.data.DataLoader(Subset(not_test_data_auged, val_index), batch_size=batch_size, shuffle=False)
            train_loader = torch.utils.data.DataLoader(Subset(not_test_data_auged, train_index), batch_size=batch_size)
        
        else:
            interval = len(not_test_index)//int(len(not_test_index)*val_split)
            val_index = not_test_index[::interval]
            val_index = np.delete(val_index, [0])
            train_index = np.setdiff1d(not_test_index, val_index)
            print(val_index.shape)
            print(train_index.shape)
            #val_index = not_test_index[:int(len(not_test_index)*val_split)]
            #train_index = not_test_index[int(len(not_test_index)*val_split):]
            val_loader = torch.utils.data.DataLoader(Subset(train_data, val_index), batch_size=batch_size, shuffle=False)
            train_loader = torch.utils.data.DataLoader(Subset(train_data, train_index), batch_size=batch_size)

    else:
        train_index = np.arange(int(2*num_ref*val_split), 2*num_ref)
        val_index = np.arange(int(2*num_ref*val_split))

        train_loader = torch.utils.data.DataLoader(Subset(train_data, train_index), batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(Subset(train_data, val_index), batch_size=batch_size, shuffle=False)
        test_loader = None

    pc_data_root = config['exp']['pc_data_root']
    pc_data_loc = pc_data_root + f'/{params["exp_name"]}'+ f'/{params["task"]}-{params["data"]["dataset"]}'

    loader_save_loc = pc_data_loc + f'/seed{seed}/id{id}-digit{digit}'
    makedirs(loader_save_loc)

    with open(loader_save_loc + '/train_loader.pkl', 'wb') as path:
        pickle.dump(train_loader, path)

    with open(loader_save_loc + '/val_loader.pkl', 'wb') as path:
        pickle.dump(val_loader, path)

    if fold_num != -1:
        with open(loader_save_loc + '/test_loader.pkl', 'wb') as path:
            pickle.dump(test_loader, path)

    return train_loader, val_loader, test_loader