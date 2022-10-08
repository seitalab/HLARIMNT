
import torch
import numpy as np
import tqdm
from typing import Tuple, Dict

from codes.data.dataset import *
from codes.supports.utils import *
from codes.supports.monitor import *
from codes.supports.storer import *
from codes.supports.min_norm_solvers import *
from codes.architectures.model import *

import sys
import yaml
import os

sys.path.append('..')

cfg_file = '../config.yaml'
with open(cfg_file) as f:
    config = yaml.safe_load(f)['exp']

class ModelTrainer:

    def __init__(
        self,
        args: Namespace,
        params: Dict,
        save_dir: str,
        seed: int,
        logger,
    ) -> None:
        """
        Args:
            args (Namespace) :
            params (Dict) :
            save_dir (str) : Directory to output results.
            seed (int) : seed of data splitting.
            logger (str) : 
        Returns:
            None 
        """
        self.args = args
        self.params = params
        self.save_dir = save_dir
        self.seed = seed
        self.logger = logger

    def run(self) -> None:
        """
        Args: 
            None
        Returns:
            None
        """
        
        digit_list = self.params['digits']
        model_cfg = self.params['model']['grouping']
        dataset_name = self.params['ref_name']
        dir_name = self.params['data_dir']

        hla_info_loc = f'codes/data/{self.params["data_dir"]}/hla_info.json'

        with open(hla_info_loc) as f:
            self.hla_info = json.load(f)
        
        save_dir = self.save_dir

        #Object to store results all over the genes.
        self.all_result_storer = AllResultStorer(save_dir)
        
        self.logger.log(f'Training of seed {self.seed} starts.')

        seed_acc_list = []
        for idx in model_cfg:
            self.all_result_storer.prepare_to_store_model_info(idx)
            hla_list_tmp = model_cfg[idx]

            for digit in digit_list:
                self.logger.log(f'Training of model {idx}-{digit} starts.')

                #Choose genes that have more than one allele.
                data_processor = DataProcessor(self.params, self.logger)
                skip_hlas, self.allele_cnts = data_processor.calc_skip_hlas(digit, hla_list_tmp)
                self.hla_list = []
                for hla in hla_list_tmp:
                    if not hla in skip_hlas:
                        self.hla_list.append(hla)
                self.all_result_storer.store_hla_list(idx, digit, self.hla_list)

                if len(skip_hlas) == len(self.hla_list):
                    self.logger.log('Skip this model because all HLA genes have only one allele.')
                    continue

                if len(skip_hlas) != 0:
                    for hla in skip_hlas:
                        self.logger.log(f'Skip {hla} because it has only one allele.')

                # Make training data and dataloader.
                train_data = data_processor.make_train_data(digit, self.hla_list) 
                
                train_loader, val_loader, test_loader = make_loaders(self.params, train_data, idx, digit, self.seed)

                #Setup model.
                input_len = train_data[0][0].shape[0]
                chunk_len = train_data[0][0][0].shape[0]
                
                self.models = implement_model(dir_name, self.hla_list, digit, input_len, chunk_len, self.params)
                self.all_result_storer.store_input_len(idx, input_len, chunk_len)

                #Transfer model of shared net.
                if digit != '2-digit':
                    self._load_model(idx, digit, save_dir + config['model_save_dir'])

                #Prepare object to store result of each gene.
                lowest_dir = '/'+str(idx)+'-'+digit
                self.storer = Storer(save_dir, lowest_dir, self.hla_list)

                #Object for early stopping.
                patience = self.params['patience']
                self.earlystopper = EarlyStopper(patience)

                #Setup Optimizer and loss function.
                model_params = []
                for key in self.models.keys():
                    self.models[key] = self.models[key].float()
                    self.models[key].train()
                    model_params += self.models[key].parameters()
                self.optimizer = torch.optim.Adam(model_params, lr=self.params['lr'])
                num_task = len(self.models) - 1
                self.loss_fn = get_loss(num_task)

                num_epoch = self.params['num_epoch']
                for epoch in range(1, num_epoch+1):
                    if epoch % self.params['decay_interval'] == 0:   
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= self.params['weight_decay']
                        self.logger.log(f'reduce learning rate at epoch {epoch}.')

                    #Objects to calculate results of each gene.
                    monitor_dict_t = {}
                    monitor_dict_v = {}
                    for hla in self.hla_list:
                        monitor_dict_t[hla] = Monitor(hla, digit)
                        monitor_dict_v[hla] = Monitor(hla, digit)
                    
                    #Train model and evaluate the accuracy.
                    acc_dict_t, monitor_dict_t = self._train(train_loader, epoch, monitor_dict_t)
                    acc_dict_v, monitor_dict_v = self._eval(val_loader, monitor_dict_v, mode='val')

                    #Store results of the epoch and amend saved model if val acc is higher than ever.
                    epoch_acc_ave_v = np.mean([acc_dict_v[key] for key in acc_dict_v.keys()])
                    epoch_acc_ave_t = np.mean([acc_dict_t[key] for key in acc_dict_t.keys()])
                    is_best = (self.earlystopper.show_best()['val_acc'] < epoch_acc_ave_v)

                    self.storer.store_epoch_result(epoch, acc_dict_t, is_eval=False)
                    self.storer.store_epoch_result(epoch, acc_dict_v, is_eval=True)
                    self.storer.amend_model(self.models, is_best)

                    #Early stopping.
                    if epoch == num_epoch:
                        self.logger.log('Stop training without early stopping.')
                        break

                    if self.earlystopper.stop_training(epoch_acc_ave_v, epoch_acc_ave_t, epoch):
                        break
                #Save results of all over the epochs.
                self.storer.save_logs()

                #Calculate the accuracy by the best model. 
                if self.params['fold_num'] != 1:
                    monitor_dict_t = {}
                    monitor_dict_v = {}
                    monitor_dict_test = {}
                    for hla in self.hla_list:
                        monitor_dict_t[hla] = Monitor(hla, digit)
                        monitor_dict_v[hla] = Monitor(hla, digit)
                        if self.params['fold_num'] != 1:
                            monitor_dict_test[hla] = Monitor(hla, digit)

                    for key in self.models.keys():
                        self.models[key].load_state_dict(torch.load(save_dir + config['model_save_dir'] + f'/{idx}-{digit}' + f'/{key}.pth'))

                    _, monitor_dict_t = self._eval(train_loader, monitor_dict_t, 'train', is_best_model=True)
                    _, monitor_dict_v = self._eval(val_loader, monitor_dict_v, 'val', is_best_model=True)
                    if self.params['fold_num'] != 1:
                        test_acc_dict, monitor_dict_test = self._eval(test_loader, monitor_dict_test, 'test', is_best_model=True)
                        if digit == self.params['digits'][-1]:
                            test_acc = [test_acc_dict[hla] for hla in test_acc_dict.keys()]
                            seed_acc_list += test_acc

        #Save  information.
        if self.params['fold_num'] != 1:
            self.all_result_storer.dump_results()
            return np.mean(seed_acc_list)

        else:
            self.all_result_storer.dump_input_len()
            return 0.0

    def _load_model(self, idx, digit, model_save_dir):
        """
        Args:
            idx:
            digit:
            model_save_dir:
        Returns:
            None 
        """
        if digit == '4-digit':
            self.models['shared'].load_state_dict(torch.load(model_save_dir + f'/{idx}-2-digit' + f'/shared.pth'))
            self.logger.log('Transferred previous digit model.')

    def _train(self, loader, epoch: int, monitor_dict: Dict) -> Tuple[float, float]:
        """
        Run one epoch of training.
        Args:
            loader:
            epoch (int):
        Returns:
            score(float):
            loss (float): 
        """

        for key in self.models.keys():
            self.models[key] = self.models[key].to(self.args.device)
            self.models[key].train()

        with tqdm.tqdm(loader) as pbar:
            pbar.set_description('[Epoch %d]' % epoch)
            for i, batch in enumerate(pbar):
                shared_input = batch[0].requires_grad_(True)
                shared_input = shared_input.to(self.args.device)

                self.labels = {}
                num_task = len(self.models) - 1
                for t in range(num_task):
                    self.labels[t] = batch[t+1]
                    self.labels[t] = self.labels[t].to(self.args.device)

                loss_data = {}
                grads = {}
                scale = {}

                self.optimizer.zero_grad()

                with torch.no_grad():
                    shared_output = self.models['shared'](shared_input.float())
                
                shared_variable = shared_output.clone().requires_grad_(True)

                for t in range(num_task):
                    self.optimizer.zero_grad()
                    hla = self.hla_list[t]
                    out_t = self.models[hla](shared_variable.float())
                    loss_t = self.loss_fn[t](out_t, self.labels[t].long())
                    loss_data[t] = loss_t.data.item()
                    loss_t.backward()
                    grads[t] = []
                    grads[t].append(shared_variable.grad.data.clone().requires_grad_(False))
                    shared_variable.grad.data.zero_()

                if num_task != 1:
                    # Normalize all gradients
                    gn = gradient_normalizers(grads, loss_data)
                    for t in range(num_task):
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]

                    # Frank-Wolfe iteration to compute scales.
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(num_task)])
                    for t in range(num_task):
                        scale[t] = float(sol[t])
                else:
                    scale[0] = 1

                # Scaled back-propagation
                self.optimizer.zero_grad()
                shared_output = self.models['shared'](shared_input.float())
                
                out = []
                for t in range(num_task):
                    hla = self.hla_list[t]
                    out_t = self.models[hla](shared_output.float())
                    out.append(out_t.float())
                    loss_t = self.loss_fn[t](out_t, self.labels[t].long())
                    loss_data[t] = loss_t.data.item()
                    if t > 0:
                        loss = loss + scale[t] * loss_t
                    else:
                        loss = scale[t] * loss_t

                loss.backward()
                self.optimizer.step()

        acc_dict_t, monitor_dict = self._eval(loader, monitor_dict, 'train')
        torch.cuda.empty_cache()

        return  acc_dict_t, monitor_dict

    def _eval(self, loader, monitor_dict_v, mode, is_best_model=False):
        torch.no_grad()
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
            if is_best_model:
                if mode == 'val':
                    self.logger.log(f'Validation accuracy of {hla} by Best Model is {acc}.')
                elif mode == 'train':
                    self.logger.log(f'Training accuracy of {hla} by Best Model is {acc}.')
                elif mode == 'test':
                    self.logger.log(f'Test accuracy of {hla}  is {acc}.')
            else:
                if mode == 'val':
                    self.logger.log(f'Validation accuracy of {hla} is {acc}.')
                elif mode == 'train':
                    self.logger.log(f'Training accuracy of {hla} is {acc}.')

        return acc_dict_v, monitor_dict_v
