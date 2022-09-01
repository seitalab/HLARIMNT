
import json
from typing import Tuple, List
from argparse import Namespace
from codes.supports.utils import *
from codes.architectures.model import implement_model
from codes.supports.storer import AllResultStorer
import pickle
from codes.supports.monitor import Monitor
import torch

cfg_file = '../config.yaml'
config = path_to_dict(cfg_file)['exp']

class ModelEvaluator:
    def __init__(self, settings) -> None:
        """
        Args:
            settings (Dict) : Settings of the experiment.
        Returns:
            None
        """
        self.settings = settings
        self.dataset_name = self.settings['ref_name']
        self.data_dir = self.settings['data_dir']
        grouping = self.settings['model']['grouping']
        self.genes = []
        for key in grouping.keys():
            self.genes += grouping[key]

    def _prepare_dataloader(self, seed, idx, digit):
        """
        Args:
            seed (int) : Fold number.
            idx (int) : Number of the model.
            digit (str)
        Returns:
            test_loader (Iterable)
        """
        loader_root = f"{self.settings['ref_name']}" + config['pc_data_dir']
        loader_file = loader_root + f'/seed{seed}/id{idx}-{digit}/test_loader.pkl'
        test_loader = pickle.load(open(loader_file, 'rb'))

        return test_loader

    def _prepare_model(self, seed, idx, digit):
        """
        Args:
            seed (int)
            idx (int)
            digit (str)
        Returns:
            models (Dict)
        """
        model_dir = f"{self.settings['ref_name']}" + config['outputs_dir'] + f'/seed{seed}' + config['model_save_dir']

        model_info_file = model_dir + '/model_info.json'
        with open(model_info_file) as f:
            model_info = json.load(f)

        self.hla_list =  model_info[str(idx)]['hla_list'][digit]
        input_len = model_info[str(idx)]['input_len']
        chunk_len = model_info[str(idx)]['chunk_len']
        models = implement_model(self.data_dir, self.hla_list, digit, input_len, chunk_len, self.settings)

        for key in models.keys():
            models[key].load_state_dict(torch.load(model_dir + f'/{idx}-{digit}' + f'/{key}.pth'))

        return models

    def _eval(self, loader, monitor_dict_v):
        """
        Args:
            loader (Iterable)
            monitor_dict_v (Dict) : Dict to hold monitors to calculate indices.
        Returns:
            acc_dict_v (Dict) : Dict to hold accuracy.
            monitor_dict_v (Dict) : Dict to hold values of indices.
        """
        with torch.no_grad():
            for m in self.models:
                self.models[m] = self.models[m].cpu()
                self.models[m].eval()

            num_task = len(self.models) - 1

            for batch in loader:
                shared_input = batch[0].requires_grad_(False)
                self.labels = {}
                for t in range(num_task):
                    self.labels[t] = batch[t+1].requires_grad_(False)

                shared_output = self.models['shared'](shared_input.float())

                out = []
                for t in range(num_task):
                    hla = self.hla_list[t]
                    out_t = self.models[hla](shared_output.float())
                    out.append(out_t.float())

                i = 0
                for hla in self.hla_list:
                    monitor_dict_v[hla].store_num_data(len(batch[0]))
                    labels_i = self.labels[i].to('cpu').detach().numpy().copy()
                    monitor_dict_v[hla].store_result(shared_input.float(), out[i], labels_i)
                    i += 1

            acc_dict_v = {}
            for hla in self.hla_list:
                acc = monitor_dict_v[hla].accuracy()
                acc_dict_v[hla] = acc
        
        return acc_dict_v, monitor_dict_v
    
    def _make_evals(self):
        """
        Calculate indices and store them.
        Args:
            None
        Returns:
            acc_dict_all (Dict) : accuracy.
        """
        model_cfg = self.settings['model']['grouping']
        digit_list = self.settings['digits']
        self.fold_num = self.settings['fold_num']

        hla_info_loc = f'codes/data/{self.settings["data_dir"]}/hla_info.json'
        with open(hla_info_loc) as f:
            hla_info = json.load(f)        
        
        acc_dict_all = {}
        for digit in digit_list:
            acc_dict_all[digit] = {}
            for hla in hla_info.keys():
                acc_dict_all[digit][hla] = []

        for seed in range(self.fold_num):
            save_dir = f'{self.settings["ref_name"]}'+  config['outputs_dir'] + f'/seed{seed}'
            self.all_result_storer = AllResultStorer(save_dir, is_test=True)

            for idx in model_cfg:
                for digit in digit_list:
                    test_loader = self._prepare_dataloader(seed, idx, digit)
                    self.models = self._prepare_model(seed, idx, digit)
                    monitor_dict = {}
                    for hla in self.hla_list:
                        monitor_dict[hla] = Monitor(hla, digit)
                    acc_dict, monitor_dict = self._eval(test_loader, monitor_dict)
                    for hla in self.hla_list:
                        acc_dict_all[digit][hla].append(acc_dict[hla])
                        evals_v = monitor_dict[hla].make_evals(hla_info)
                        self.all_result_storer.store_evals(evals_v)

            self.all_result_storer.dump_results()
        return acc_dict_all
                    
    def run(self):
        acc_dict_all = self._make_evals()
