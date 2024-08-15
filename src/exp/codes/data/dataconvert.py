
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from codes.supports.utils import *
import pickle
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data.dataset import Subset

config_file = '../config.yaml'
config = path_to_dict(config_file)

class DataAugumentor:

    def __init__(self, params, model_id, digit):
        """
        Args:
            data (np.ndarray): 1(cls token)をつけた後のid変換済みデータ
        """
        self.params = params
        self.model_id = model_id
        self.digit = digit

    def _calc_label_num(self, not_test_data):
        """
        """
        dataset_name = self.params['data']['dataset']
        data_loc = config['dataset']
        hla_info = data_loc['save_root'] + data_loc[dataset_name]['dirname'] + data_loc[dataset_name]['hla_info']
        hla_info = path_to_dict(hla_info)

        label_num = len(hla_info[self.params['model']['grouping'][self.model_id][0]][self.digit])

        data_num_by_label = [0 for _ in range(label_num)]
        for sample in not_test_data:
            label = int(sample[1])
            data_num_by_label[label] += 1

        return np.array(data_num_by_label)

    def _random_crop(self, sample, aug_num):
        """
        """
        rest_rate = self.params['rest_rate']
        auged_samples = [sample]
        for aug in range(aug_num - 1):
            rest_rates = torch.full(torch.tensor(sample[0]).size(),fill_value=rest_rate)
            cropper = torch.bernoulli(rest_rates)
            cropped_sample = torch.tensor(sample[0]) * cropper
            if not self.params['encode'] == 'chunk':
                cropped_sample[0] = 1
            tmp = [np.array(cropped_sample)]
            for i in sample[1:]:
                tmp.append(i)
            auged_samples.append(tmp)
        return auged_samples

    def augument(self, not_test_data):
        """
        """
        assert self.params['data_aug'] == True
        if self.params['aug_even']:
            data_num = len(not_test_data)
            data_num_by_label = self._calc_label_num(not_test_data)
            auged_data_num = data_num * self.params['aug_rate']
            label_kind_num = np.shape(data_num_by_label[np.nonzero(data_num_by_label)])[0]

            final_samples = []
            for sample in not_test_data:
                label = int(sample[1])
                auged_data_num_by_label = auged_data_num // label_kind_num
                auged_data_num_by_sample = auged_data_num_by_label // data_num_by_label[label]
                auged_samples = self._random_crop(sample, auged_data_num_by_sample)
                final_samples += auged_samples
            return final_samples

        else:
            data_num = len(not_test_data)
            data_num_by_label = self._calc_label_num(not_test_data)
            auged_data_num = data_num * self.params['aug_rate']
            label_kind_num = np.shape(data_num_by_label[np.nonzero(data_num_by_label)])[0]

            final_samples = []
            for sample in not_test_data:
                label = int(sample[1])
                auged_data_num_by_label = data_num_by_label[label] * self.params['aug_rate']
                auged_data_num_by_sample = auged_data_num_by_label // data_num_by_label[label]
                auged_samples = self._random_crop(sample, auged_data_num_by_sample)
                final_samples += auged_samples

            return final_samples                
