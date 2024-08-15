
import json
from typing import Tuple, List
from argparse import Namespace
from codes.supports.utils import *
from codes.architectures.model import implement_model
from codes.supports.storer import AllResultStorer
import pickle
from codes.supports.monitor import Monitor

import torch
import matplotlib.pyplot as plt

from codes.train_model import ModelTrainer

cfg_file = '../config.yaml'
config = path_to_dict(cfg_file)

class ModelEvaluator:
    def __init__(self, args) -> None:
        """
        Args:
            args (Namespace):
            seed (int) : Express the part of test data.
        """
        self.args = args
        self.settings = path_to_dict(args.settings)['base']
        self.dataset_name = self.settings['data']['dataset']
        grouping = self.settings['model']['grouping']
        self.genes = []
        for key in grouping.keys():
            self.genes += grouping[key]
        self.sample_result = ['R2', 'accuracy']

        self.exp_dir = f'/{self.settings["exp_name"]}'+ f'/{self.settings["task"]}-{self.settings["data"]["dataset"]}'

    def _prepare_dataloader(self, seed, idx, digit):
        """
        Args:
            None
        Returns:
            test_loader (Iterable)
        """
        loader_root = config['exp']['pc_data_root'] + self.exp_dir
        loader_file = loader_root + f'/seed{seed}/id{idx}-digit{digit}/test_loader.pkl'
        test_loader = pickle.load(open(loader_file, 'rb'))

        return test_loader

    def _prepare_model(self, seed, idx, digit):
        """
        Args:
            None
        Returns:
            None
        """
        model_dir = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['model_save_loc']

        model_info_file = model_dir + '/model_info.json'
        with open(model_info_file) as f:
            model_info = json.load(f)

        self.hla_list =  model_info[str(idx)]['hla_list'][digit]
        input_len = model_info[str(idx)]['input_len'] ###(trainで保存) {1(idx):{input_len:25, hla_list:{2-digit:[], 4-digit:[]}} }
        if self.settings['encode'] == 'chunk':
            chunk_len = model_info[str(idx)]['chunk_len']
        else:
            chunk_len = None
        models = implement_model(self.dataset_name, self.hla_list, digit, input_len, chunk_len, self.settings)

        for key in models.keys():
            models[key].load_state_dict(torch.load(model_dir + f'/{idx}-{digit}' + f'/{key}.pth'))

        return models

    def _eval(self, loader, monitor_dict_v):
        with torch.no_grad():
            #mc_drop_num = 100
            #for i in range(mc_drop_num):
            for m in self.models:
                self.models[m] = self.models[m].cpu()
                self.models[m].eval()#.train()

            num_task = len(self.models) - 1

            for batch in loader:
                shared_input = batch[0].requires_grad_(False)
                self.labels = {}
                for t in range(num_task):
                    self.labels[t] = batch[t+1].requires_grad_(False)
                mask_input = None
                mask_conv1 = None
                mask_conv2 = None
                filters = {'mask_input':mask_input, 'mask_conv1':mask_conv1, 'mask_conv2':mask_conv2}
                shared_output, filters = self.models['shared'](shared_input.float(), filters)

                out = []
                for t in range(num_task):
                    hla = self.hla_list[t]
                    out_t, mask_t = self.models[hla](shared_output.float(), {'mask_fc':None})
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
                #print('貯められたデータは',monitor_dict_v[hla].num_data)
                acc_dict_v[hla] = acc
        
        return acc_dict_v, monitor_dict_v
    
    def _make_evals(self):
        model_cfg = self.settings['model']['grouping']
        digit_list = self.settings['digits']
        self.fold_num = self.settings['fold_num']
        data_loc = config['dataset']
        dataset_name = self.settings['data']['dataset']
        hla_info = data_loc['save_root'] + data_loc[dataset_name]['dirname'] + data_loc[dataset_name]['hla_info']
        hla_info = path_to_dict(hla_info)
        freq_info_loc = data_loc[dataset_name]['allele_freq_info']
        #freq_info = pd.read_table(data_loc['save_root'] + data_loc[dataset_name]['dirname'] + freq_info_loc, sep='\t|\s+',engine='python', index_col=1)

        if dataset_name == 'Mixed' or dataset_name == 'Equal' or dataset_name == 'Ind_Pan-Asian' or dataset_name == 'Ind_T1DGC'or dataset_name == 'T1DGC_530' or dataset_name == 'T1DGC_1300' or dataset_name == 'T1DGC_2600':
            freq_info = pd.read_csv(data_loc['save_root'] + data_loc[dataset_name]['dirname'] + freq_info_loc ,index_col=1)
        else:
            freq_info = pd.read_table(data_loc['save_root'] + data_loc[dataset_name]['dirname'] + freq_info_loc, sep='\t|\s+',engine='python', index_col=1)
        
        acc_dict_all = {}
        for digit in digit_list:
            acc_dict_all[digit] = {}
            for hla in hla_info.keys():
                acc_dict_all[digit][hla] = []

        ##########
        if self.fold_num == -1:
            seed_num = 1
        else:
            seed_num = self.fold_num
        for seed in range(seed_num):
            self.all_result_storer = AllResultStorer(config['exp']['save_root'] + self.exp_dir + f'/seed{seed}', is_test=True)

            for idx in model_cfg:
                for digit in digit_list:
                    test_loader = self._prepare_dataloader(seed, idx, digit)
                
                    self.models = self._prepare_model(seed, idx, digit)
                    #self.models.train()
                    monitor_dict = {}
                    for hla in self.hla_list:
                        monitor_dict[hla] = Monitor(hla, digit)
                    acc_dict, monitor_dict = self._eval(test_loader, monitor_dict)
                    for hla in self.hla_list:
                        acc_dict_all[digit][hla].append(acc_dict[hla])
                        id_num = len(hla_info[hla][digit])
                        freqs = [freq_info.at[allele, 'MAF'] for allele in hla_info[hla][digit]]
                        evals_by_freq_v, sample_result_v = monitor_dict[hla].make_evals_by_freq(id_num, freqs)
                        self.all_result_storer.store_evals_by_freq(evals_by_freq_v, sample_result_v, 'test')

            self.all_result_storer.dump_results()
        return acc_dict_all
                    
    def _make_freq_gene_evals(self):
        if self.fold_num == -1:
            seed_num = 1
        else:
            seed_num = self.fold_num

        freq_gene_info = {}
        for criteria in ['ppv', 'sens', 'fscore', 'r2','r2_01','concordance', 'R2', 'accuracy', 'confidence']:
            freq_gene_info[criteria] = {}
            for hla in self.genes:
                freq_gene_info[criteria][hla] = {}
                for digit in ['2-digit', '4-digit']:
                    freq_gene_info[criteria][hla][digit] = {}
                    i = 0
                    next_phase = self.args.phases[i+1]
                    for phase in self.args.phases:
                        freq_gene_info[criteria][hla][digit][phase] = {}
                        freq_gene_info[criteria][hla][digit][phase]['value'] = []
                        freq_gene_info[criteria][hla][digit][phase]['count'] = []
                        for seed in range(seed_num):
                            result_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_test.csv'
                            sample_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_sample_test.csv'
                            if criteria in self.sample_result:
                                result_df = pd.read_csv(sample_csv)
                            else:
                                result_df = pd.read_csv(result_csv)
                            tmp_df_2 = result_df[result_df['digit']==digit]
                            number_of_allele = len(tmp_df_2[(tmp_df_2['freq']>=phase)&(tmp_df_2['freq']<next_phase)&(tmp_df_2['hla']==hla)][criteria].tolist())
                            #if number_of_allele == 0:
                                #score_by_seed_phase_2 = None
                            #else:
                            score_by_seed_phase_2 = np.nanmean(tmp_df_2[(tmp_df_2['freq']>=phase)&(tmp_df_2['freq']<next_phase)&(tmp_df_2['hla']==hla)][criteria].tolist())
                            freq_gene_info[criteria][hla][digit][phase]['value'].append(score_by_seed_phase_2)
                            freq_gene_info[criteria][hla][digit][phase]['count'].append(number_of_allele)
                        if i == len(self.args.phases)-2 or i == len(self.args.phases)-1:
                            next_phase = 1.0
                        else:
                            next_phase = self.args.phases[i+2]
                        i += 1
        return freq_gene_info
                            
    def _make_phased_evals(self):
        if self.fold_num == -1:
            seed_num = 1
        else:
            seed_num = self.fold_num
        phased_result = {}
        for criteria in ['ppv', 'sens', 'fscore', 'r2','r2_01','concordance', 'R2', 'accuracy', 'confidence']:
            phased_result_tmp = {'2-digit':{}, '4-digit':{}}
            i = 0
            next_phase = self.args.phases[i+1]
            for phase in self.args.phases:
                phased_result_tmp['2-digit'][phase] = []
                phased_result_tmp['4-digit'][phase] = []
                for seed in range(seed_num):
                    result_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_test.csv'
                    sample_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_sample_test.csv'
                    if criteria in self.sample_result:
                        result_df = pd.read_csv(sample_csv)
                    else:
                        result_df = pd.read_csv(result_csv)
                    tmp_df_2 = result_df[result_df['digit']=='2-digit']
                    tmp_df_4 = result_df[result_df['digit']=='4-digit']
                    #if number_of_allele_2 == 0:
                        #score_by_seed_phase_2 = None
                    #else:
                    score_by_seed_phase_2 = np.nanmean(tmp_df_2[(tmp_df_2['freq']>=phase)&(tmp_df_2['freq']<next_phase)][criteria].tolist())
                    score_by_seed_phase_4 = np.nanmean(tmp_df_4[(tmp_df_4['freq']>=phase)&(tmp_df_4['freq']<next_phase)][criteria].tolist())
                    #score_by_seed_phase_2 = np.mean(tmp_df_2[tmp_df_2['freq']<phase][criteria].tolist())
                    #score_by_seed_phase_4 = np.mean(tmp_df_4[tmp_df_4['freq']<phase][criteria].tolist())
                    phased_result_tmp['2-digit'][phase].append(score_by_seed_phase_2)
                    phased_result_tmp['4-digit'][phase].append(score_by_seed_phase_4)
                if i == len(self.args.phases)-2 or i == len(self.args.phases)-1:
                    next_phase = 1.0
                else:
                    next_phase = self.args.phases[i+2]
                i += 1
            phased_result[criteria] = phased_result_tmp

        return phased_result

    def _make_gene_evals(self):
        if self.fold_num == -1:
            seed_num = 1
        else:
            seed_num = self.fold_num
        phased_result = {}
        for criteria in ['ppv', 'sens', 'fscore', 'r2','r2_01','concordance', 'R2', 'accuracy','confidence']:
            phased_result_tmp = {'2-digit':{}, '4-digit':{}}
            for hla in self.genes:
                phased_result_tmp['2-digit'][hla] = []
                phased_result_tmp['4-digit'][hla] = []
                for seed in range(seed_num):
                    result_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_test.csv'
                    sample_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_sample_test.csv'
                    if criteria in self.sample_result:
                        result_df = pd.read_csv(sample_csv)
                    else:
                        result_df = pd.read_csv(result_csv)
                    tmp_df_2 = result_df[result_df['digit']=='2-digit']
                    tmp_df_4 = result_df[result_df['digit']=='4-digit']
                    number_of_allele_2 = len(tmp_df_2[tmp_df_2['hla']==hla][criteria].tolist())
                    number_of_allele_4 = len(tmp_df_4[tmp_df_4['hla']==hla][criteria].tolist())
                    #if number_of_allele_2 == 0:
                        #score_by_seed_phase_2 = None
                    #else:
                    score_by_seed_phase_2 = np.nanmean(tmp_df_2[tmp_df_2['hla']==hla][criteria].tolist())
                    #if number_of_allele_4 == 0:
                        #score_by_seed_phase_4 = None
                    #else:
                    score_by_seed_phase_4 = np.nanmean(tmp_df_4[tmp_df_4['hla']==hla][criteria].tolist())
                    #score_by_seed_phase_2 = np.mean(tmp_df_2[tmp_df_2['freq']<phase][criteria].tolist())
                    #score_by_seed_phase_4 = np.mean(tmp_df_4[tmp_df_4['freq']<phase][criteria].tolist())
                    phased_result_tmp['2-digit'][hla].append(score_by_seed_phase_2)
                    phased_result_tmp['4-digit'][hla].append(score_by_seed_phase_4)
            phased_result[criteria] = phased_result_tmp
        return phased_result

    def _make_evals_for_scatter(self):
        data_loc = config['dataset']
        dataset_name = self.settings['data']['dataset']
        hla_info = data_loc['save_root'] + data_loc[dataset_name]['dirname'] + data_loc[dataset_name]['hla_info']
        hla_info = path_to_dict(hla_info)
        freq_info_loc = data_loc[dataset_name]['allele_freq_info']
        freq_info = pd.read_table(data_loc['save_root'] + data_loc[dataset_name]['dirname'] + freq_info_loc, sep='\t|\s+',engine='python', index_col=1)

        results = {'hla':[], 'digit':[],'r2':[], 'r2_01':[],'ppv':[], 'sens':[], 'fscore':[], 'freq':[], 'id':[]}
        for id in range(id_num):
            for digit in self.settings['digits']:
                for hla in self.genes:
                    id_num = len(hla_info[hla][digit])
                    cri_result = {}
                    freqs = [freq_info.at[allele, 'MAF'] for allele in hla_info[hla][digit]]
                    freq = freqs[id]
                    for criteria in ['ppv', 'sens', 'fscore', 'r2', 'r2_01','concordance']:
                        value_by_seed = []
                        for seed in range(self.fold_num):
                            result_csv = config['exp']['save_root'] + self.exp_dir + f'/seed{seed}' + config['exp']['csv_save_loc'] + '/evals_by_freq_test.csv'
                            result_df = pd.read_csv(result_csv)
                            tmp_df = result_df[result_df['digit']==digit]
                            tmp_df = tmp_df[tmp_df['id']==id]
                            score_by_seed_phase = np.nanmean(tmp_df_4[tmp_df_4['hla']==hla][criteria].tolist())
                            value_by_seed.append(score_by_seed_phase)
                        value = np.nanmean(value_by_seed)
                        cri_result[criteria] = value

                    results['hla'].append(hla)
                    results['digit'].append(digit)
                    results['freq'].append(freq)
                    results['id'].append(id)
                    for key in cri_result.keys():
                        results[key].append(cri_result[key])

        return results

    def run(self):
        acc_dict_all = self._make_evals()
        if not self.args.density:
            details = self._make_freq_gene_evals()
            phased_result = self._make_phased_evals()
            gene_result = self._make_gene_evals()
            #scatter_results = self._make_evals_for_scatter()
            #details = self._make_freq_gene_evals()
            eval_result_dir = config['exp']['save_root'] + self.exp_dir

            with open(eval_result_dir +  '/test_phased_evals.json', 'w') as f:
                json.dump(phased_result, f, indent=4)
        
            with open(eval_result_dir + '/test_gene_evals.json', 'w') as f:
                json.dump(gene_result, f, indent=4)

            with open(eval_result_dir +  '/test_details.json', 'w') as f:
                json.dump(details, f, indent=4)  
            
            #df = pd.DataFrame(scatter_results)
            #df.to_csv(eval_result_dir + f'/test_scatter.csv')
            
            results_2 = {}
            results_4 = {}
            results_gene_2 = {}
            results_gene_4 = {}

            for criteria in ['ppv', 'sens', 'fscore', 'r2','r2_01','concordance', 'R2', 'accuracy','confidence']:
                results_2[criteria] = {}
                results_4[criteria] = {}
                results_gene_2[criteria] = {}
                results_gene_4[criteria] = {}
                for phase in self.args.phases:
                    #if not phased_result[criteria]['2-digit'][phase] is None:
                    result_tmp_2 = np.nanmean(phased_result[criteria]['2-digit'][phase])
                    #else:
                    #result_tmp_2 = None
                    #if not phased_result[criteria]['4-digit'][phase] is None:
                    result_tmp_4 = np.nanmean(phased_result[criteria]['4-digit'][phase])
                    #else:
                        #result_tmp_4 = None
                    results_2[criteria][phase] = result_tmp_2
                    results_4[criteria][phase] = result_tmp_4
                for gene in self.genes:
                    result_tmp_gene_2 = np.nanmean(gene_result[criteria]['2-digit'][gene])
                    result_tmp_gene_4 = np.nanmean(gene_result[criteria]['4-digit'][gene])
                    results_gene_2[criteria][gene] = result_tmp_gene_2
                    results_gene_4[criteria][gene] = result_tmp_gene_4
            """
            for criteria in ['R2', 'accuracy']:
                sample_2[criteria] = {}
                sample_4[criteria] = {}
                sample_gene_2[criteria] = {}
                sample_gene_4[criteria] = {}
                for phase in self.args.phases:
                    #if not phased_result[criteria]['2-digit'][phase] is None:
                    result_tmp_2 = np.mean(sample_phased_result[criteria]['2-digit'][phase])
                    #else:
                    #result_tmp_2 = None
                    #if not phased_result[criteria]['4-digit'][phase] is None:
                    result_tmp_4 = np.mean(sample_phased_result[criteria]['4-digit'][phase])
                    #else:
                        #result_tmp_4 = None
                    sample_2[criteria][phase] = result_tmp_2
                    sample_4[criteria][phase] = result_tmp_4
                for gene in self.genes:
                    result_tmp_gene_2 = np.mean(gene_result[criteria]['2-digit'][gene])
                    result_tmp_gene_4 = np.mean(gene_result[criteria]['4-digit'][gene])
                    sample_gene_2[criteria][gene] = result_tmp_gene_2
                    sample_gene_4[criteria][gene] = result_tmp_gene_4
                    """

            with open(eval_result_dir +  '/test_phased_evals_mean_2digit.json', 'w') as f:
                json.dump(results_2, f, indent=4)
            with open(eval_result_dir +  '/test_phased_evals_mean_4digit.json', 'w') as f:
                json.dump(results_4, f, indent=4)  
            with open(eval_result_dir +  '/test_genes_evals_mean_2digit.json', 'w') as f:
                json.dump(results_gene_2, f, indent=4)
            with open(eval_result_dir +  '/test_genes_evals_mean_4digit.json', 'w') as f:
                json.dump(results_gene_4, f, indent=4)              
            with open(eval_result_dir +  '/acc_2digit.json', 'w') as f:
                json.dump(acc_dict_all['2-digit'], f, indent=4)            
            with open(eval_result_dir +  '/acc_4digit.json', 'w') as f:
                json.dump(acc_dict_all['4-digit'], f, indent=4)            

    def _store_eval_curve(self, results, eval_result_dir):
        """
        Args:
            None
        Returns:
            None
        """
        for criteria in results:
            fig, ax = plt.subplots()
            ax.plot([i for i in range(len(self.args.phases))], [results[criteria][phase] for phase in self.args.phases], label=criteria)
            plt.xticks([i for i in range(len(self.args.phases))], self.args.phases)
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            plt.savefig(eval_result_dir + f'/{criteria}_by_allele_freq.png')

class ScatterStorer:

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

    def csv2scatter(self, save_dir, digit):
        assert len(self.kwargs) == 2
        for criteria in ['ppv', 'sens', 'fscore', 'r2','r2_01']:
            ###fig準備
            return


class ComparedResultStorer:

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs : Expected to be transformer='path/to/transformer/result/.json',...
        """
        self.kwargs = kwargs.copy()

    def _file_to_dict(self):
        result_dict = {}
        for key in self.kwargs.keys():
            path = self.kwargs[key]
            with open(path) as f:
                dict_tmp = json.load(f)
                result_dict[key] = dict_tmp
        return result_dict

    def _convert_dict(self, result_dict):
        """
        {ppv:{transformer:{0.001:1.0,...}}}にする
        {dense0.5:{ppv:1.0, ...}} -> {ppv:{dense0.5:}}
        """
        results = {}
        keys = result_dict[list(result_dict.keys())[0]].keys() ###criteriaごとに比較するグラフが変わるならだめ

        for criteria in keys:
            results[criteria] = {}
            for net in result_dict.keys():
                results[criteria][net] = result_dict[net][criteria]

        return results

    def store_eval_curve(self, save_dir, digit, type):
        """ 
        """
        results = self._file_to_dict()
        results = self._convert_dict(results)
        if digit == '2digit':
            digit_ = '2-digit'
        elif digit == '4digit':
            digit_ = '4-digit'

        with open('/root/export/users/kaho/prime_hla/exp/results/exp5_Equal/prime_hla-Equal/test_details.json') as f:#####ここ変えろよ！！！！！
            prime_details = json.load(f)
        with open('/root/export/users/kaho/prime_hla/exp/results/exp5_Equal/deep_hla-Equal/test_details.json') as f:#####ここ変えろよ！！！！！
            deep_details = json.load(f)
        mittume = False
        mittume_path = '/root/export/users/kaho/prime_hla/exp/results/exp18_Pan-Asian/prime_hla-Pan-Asian/test_details.json' #######変えろよ！！！！！
        if os.path.exists(mittume_path):
            mittume = True
            with open(mittume_path) as f:
                mittume_details = json.load(f)

        detail_dict = {}
        linestyle = ['solid','dashed']

        if type == 'genes':
            freq_list = ['all','rare']
        else:
            freq_list = ['phase']
        for all_or_001 in freq_list:
            for criteria in results:
                detail_dict[criteria] = {}
                prime_details_tmp = prime_details[criteria]
                deep_details_tmp = deep_details[criteria]
                if mittume:
                    mittume_details_tmp = mittume_details[criteria]
                fig, ax = plt.subplots()
                if criteria == 'sens':
                    graph_title = 'sensitivity'
                elif criteria == 'ppv':
                    graph_title = 'PPV'
                elif criteria == 'r2':
                    graph_title = '$r^2$'
                elif criteria == 'confidence':
                    graph_title = 'probability'
                else:
                    graph_title = criteria
                plt.title(f'{digit_} allele', fontsize=20)
                
                i = 0
                for net in results[criteria]:
                    phases = [key for key in results[criteria][net].keys()]
                    if net == 'prev_research':
                        label = 'DEEP*HLA'
                    elif net == 'Transformer':
                        label = 'HLARIMNT'
                    elif net == 'rnn':
                        label = 'RNN'
                    elif net == 'all_chunk_tf':
                        label = 'Using All SNPs'

                    if type == 'phased':
                        left = np.arange(len(phases))
                        height = -0.3
                        if label == 'DEEP*HLA':
                            color_ = 'C0'
                            pos = left
                        if label == 'HLARIMNT':
                            color_ = 'C1'
                            pos = left + height
                        if label == 'Using All SNPs':
                            color_ = 'C2'
                            pos = left + height
                        if label == 'RNN':
                            color_ = 'C2'
                            pos = left + height
                        #ax.plot([i for i in range(len(phases))], [results[criteria][net][phase] for phase in phases], label=label, marker='o', linestyle=linestyle[i%2])
                        ax.barh(pos,[results[criteria][net][phase] for phase in phases] , align="center", height=height, label=label,color=color_)
                    i += 1
                if type == 'phased':
                    phases_memori = [f"{j}~" for j in phases]
                    phases_memori[0] = '<0.005'
                elif type == 'genes':
                    phases_memori = [i for i in phases]

                if type == 'phased':
                    ax.set_xlim(0.0, 1.0)
                    ax.set_ylabel('Allele frequency',fontsize=20)
                if type == 'genes':
                    ax.set_xlim(0.0, 1.0)
                    #ax.set_ylabel('HLA',fontsize=20)
                    #ax2.set_ylim(0.0, 1.0)
                    dict_for_bar = {}
                    dict_for_bar_deep = {}
                    dict_for_bar_mittume = {}

                    for hla in phases:
                        if all_or_001 == 'rare':
                            dict_for_bar[hla] = np.nanmean(prime_details[criteria][hla][digit_]['0.0']['value']*prime_details[criteria][hla][digit_]['0.0']['count'][0]+prime_details[criteria][hla][digit_]['0.005']['value']*prime_details[criteria][hla][digit_]['0.005']['count'][0])
                            dict_for_bar_deep[hla] = np.nanmean(deep_details[criteria][hla][digit_]['0.0']['value']*deep_details[criteria][hla][digit_]['0.0']['count'][0]+deep_details[criteria][hla][digit_]['0.005']['value']*deep_details[criteria][hla][digit_]['0.005']['count'][0])
                            if mittume:
                                dict_for_bar_mittume[hla] = np.nanmean(mittume_details[criteria][hla][digit_]['0.0']['value']*mittume_details[criteria][hla][digit_]['0.0']['count'][0]+mittume_details[criteria][hla][digit_]['0.005']['value']*mittume_details[criteria][hla][digit_]['0.005']['count'][0])
                        elif all_or_001 == 'all':
                            dict_for_bar[hla] = np.nanmean(prime_details[criteria][hla][digit_]['0.0']['value']*prime_details[criteria][hla][digit_]['0.0']['count'][0]+\
                            prime_details[criteria][hla][digit_]['0.005']['value']*prime_details[criteria][hla][digit_]['0.005']['count'][0]+\
                            prime_details[criteria][hla][digit_]['0.01']['value']*prime_details[criteria][hla][digit_]['0.01']['count'][0]+\
                            prime_details[criteria][hla][digit_]['0.05']['value']*prime_details[criteria][hla][digit_]['0.05']['count'][0]+\
                            prime_details[criteria][hla][digit_]['0.1']['value']*prime_details[criteria][hla][digit_]['0.1']['count'][0])
        
                            dict_for_bar_deep[hla] = np.nanmean(deep_details[criteria][hla][digit_]['0.0']['value']*deep_details[criteria][hla][digit_]['0.0']['count'][0]+\
                            deep_details[criteria][hla][digit_]['0.005']['value']*deep_details[criteria][hla][digit_]['0.005']['count'][0]+\
                            deep_details[criteria][hla][digit_]['0.01']['value']*deep_details[criteria][hla][digit_]['0.01']['count'][0]+\
                            deep_details[criteria][hla][digit_]['0.05']['value']*deep_details[criteria][hla][digit_]['0.05']['count'][0]+\
                            deep_details[criteria][hla][digit_]['0.1']['value']*deep_details[criteria][hla][digit_]['0.1']['count'][0])

                            if mittume:
                                dict_for_bar_mittume[hla] = np.nanmean(mittume_details[criteria][hla][digit_]['0.0']['value']*mittume_details[criteria][hla][digit_]['0.0']['count'][0]+\
                                mittume_details[criteria][hla][digit_]['0.005']['value']*mittume_details[criteria][hla][digit_]['0.005']['count'][0]+\
                                mittume_details[criteria][hla][digit_]['0.01']['value']*mittume_details[criteria][hla][digit_]['0.01']['count'][0]+\
                                mittume_details[criteria][hla][digit_]['0.05']['value']*mittume_details[criteria][hla][digit_]['0.05']['count'][0]+\
                                mittume_details[criteria][hla][digit_]['0.1']['value']*mittume_details[criteria][hla][digit_]['0.1']['count'][0])

                    prime_bar = [dict_for_bar[hla_] for hla_ in ['HLA_A','HLA_C','HLA_B','HLA_DRB1','HLA_DQA1','HLA_DQB1','HLA_DPA1','HLA_DPB1']]
                    deep_bar = [dict_for_bar_deep[hla_] for hla_ in ['HLA_A','HLA_C','HLA_B','HLA_DRB1','HLA_DQA1','HLA_DQB1','HLA_DPA1','HLA_DPB1']]
                    if mittume:
                        mittume_bar = [dict_for_bar_mittume[hla_] for hla_ in ['HLA_A','HLA_C','HLA_B','HLA_DRB1','HLA_DQA1','HLA_DQB1','HLA_DPA1','HLA_DPB1']]

                    detail_dict[criteria]['prime'] = prime_bar
                    detail_dict[criteria]['deep'] = deep_bar
                    #if mittume:
                        #detail_dict[criteria]['mittume'] = mittume_bar

                    #ax.barh(phases, prime_bar, align="center", height=0.3, label='HLARIMNT')
                    #ax.barh(phases, deep_bar, align="center", height=0.3, label='DEEP*HLA')
                    left = np.arange(len(prime_bar))
                    height = -0.3
                    
                    ax.barh(left, deep_bar, align="center", height=height, label='DEEP*HLA',color='C0')
                    ax.barh(left+height, prime_bar, align="center", height=height, label='HLARIMNT',color='C1')

                if type == 'genes':
                    phases_memori = ['HLA_A','HLA_C','HLA_B','HLA_DRB1','HLA_DQA1','HLA_DQB1','HLA_DPA1','HLA_DPB1']
                    plt.yticks(left + height/2, phases_memori)
                    ax.set_xlabel(graph_title, fontsize=20)
                else:
                    plt.yticks(np.arange(len(phases))+height/2, phases_memori)
                    ax.set_xlabel(graph_title, fontsize=20)
                plt.tick_params(labelsize=16)

                #if type == 'phased':
                    #ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=3,fontsize=12)
                #else:
                    #ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2,fontsize=12)

                plt.tight_layout()
                plt.savefig(save_dir + f'/{all_or_001}_{criteria}_compare_{type}_evals_mean_{digit}.png')
            with open(f'/root/export/users/kaho/prime_hla/exp/results/exp5_Equal/{all_or_001}detail_bar_{digit}.json', 'w') as f:###ここも変えろよ！！！！！
                json.dump(detail_dict, f, indent=4)

    def store_acc_curve(self, save_dir, digit):

        results = self._file_to_dict()
        #results = self._convert_dict(results)
        
        fig, ax = plt.subplots()
        plt.title(f'accuracy of {digit} allele')
        linestyle = ['dashed','solid']
        i = 0
        for net in results:
            genes = [key for key in results[net].keys()]
            accuracy = [np.mean(results[net][hla]) for hla in genes]
            ax.plot([i for i in range(len(genes))], accuracy, label=net, marker='D', linestyle=linestyle[i%2])
            i += 1
        plt.xticks([i for i in range(len(genes))], genes)
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        plt.savefig(save_dir + f'/accuracy_{digit}.png')
