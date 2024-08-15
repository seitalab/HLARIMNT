
from codes.eval_model import ModelEvaluator, ComparedResultStorer
from argparse import Namespace
from codes.supports.utils import path_to_dict

def run_eval(args: Namespace) -> None:
    """
    """
    cfg_file = '../config.yaml'
    config = path_to_dict(cfg_file)
    #dataset_name = settings['data']['dataset']
    exp_dir = f'/{args.exp_name}'
    #exp_dir = f'/{settings["exp_name"]}'#+ f'/{settings["task"]}-{settings["data"]["dataset"]}'
    save_dir = config['exp']['save_root'] + exp_dir

    #settings = path_to_dict(args.settings)['base']
    if not args.compare:
        evaluator = ModelEvaluator(args)
        evaluator.run()
    elif not args.acc:
        #####
        for type in ['phased', 'genes']:
            for digit in ['4digit']:

                # exp5_default_t1dgc =  config['exp']['save_root'] + '/exp5_T1DGC' + '/deep_hla-T1DGC' + f'/test_{type}_evals_mean_{digit}.json'
                # exp5_default_Pan_Asian =  config['exp']['save_root'] + '/exp5_Pan-Asian' + '/deep_hla-Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'

                # exp5_tf_t1dgc =  config['exp']['save_root'] + '/exp5_T1DGC' + '/prime_hla-T1DGC' + f'/test_{type}_evals_mean_{digit}.json'
                # exp5_tf_Pan_Asian =  config['exp']['save_root'] + '/exp5_Pan-Asian' + '/prime_hla-Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'

                exp14_default_t1dgc =  config['exp']['save_root'] + '/exp14_T1DGC' + '/deep_hla-T1DGC' + f'/test_{type}_evals_mean_{digit}.json'
                exp14_default_Pan_Asian =  config['exp']['save_root'] + '/exp14_Pan-Asian' + '/deep_hla-Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'
                # rnn = config['exp']['save_root'] + '/exp12_Pan-Asian' + '/rnn-Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'

                exp14_tf_t1dgc =  config['exp']['save_root'] + '/exp14_T1DGC' + '/prime_hla-T1DGC' + f'/test_{type}_evals_mean_{digit}.json'
                exp14_tf_Pan_Asian =  config['exp']['save_root'] + '/exp14_Pan-Asian' + '/prime_hla_retry-Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'

                # tf_Ind_Pan_Asian = config['exp']['save_root'] + '/exp11_Ind_Pan-Asian' + '/prime_hla-Ind_Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'
                # default_Ind_Pan_Asian = config['exp']['save_root'] + '/exp11_Ind_Pan-Asian' + '/deep_hla-Ind_Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'

                # tf_Ind_T1DGC = config['exp']['save_root'] + '/exp11_Ind_T1DGC' + '/prime_hla-Ind_T1DGC' + f'/test_{type}_evals_mean_{digit}.json'
                # default_Ind_T1DGC = config['exp']['save_root'] + '/exp11_Ind_T1DGC' + '/deep_hla-Ind_T1DGC' + f'/test_{type}_evals_mean_{digit}.json'

                # tf_mixed = config['exp']['save_root'] + '/exp5_Mixed' + '/prime_hla-Mixed' + f'/test_{type}_evals_mean_{digit}.json'
                # default_mixed = config['exp']['save_root'] + '/exp5_Mixed' + '/deep_hla-Mixed' + f'/test_{type}_evals_mean_{digit}.json'

                # tf_equal = config['exp']['save_root'] + '/exp5_Equal' + '/prime_hla-Equal' + f'/test_{type}_evals_mean_{digit}.json'
                # default_equal = config['exp']['save_root'] + '/exp5_Equal' + '/deep_hla-Equal' + f'/test_{type}_evals_mean_{digit}.json'

                
                all = config['exp']['save_root'] + '/exp13_Pan-Asian' + '/prime_hla_all-Pan-Asian' + f'/test_{type}_evals_mean_{digit}.json'
                storer = ComparedResultStorer(Transformer=tf_equal, prev_research=default_equal)

                #####
                storer.store_eval_curve(save_dir, digit, type)
    # else:
    #     for digit in ['2digit', '4digit']:
    #         transformer_500w = save_dir + '/noDA-by_input_len_500w-T1DGC' + f'/acc_{digit}.json'
    #         transformer_100w = save_dir + '/noDA-by_input_len_100w-T1DGC' + f'/acc_{digit}.json'
    #         #ver2 = save_dir + '/noDA-by_input_len_70w_2-Pan-Asian' + f'/test_phased_evals_mean_{digit}.json'
    #         cnn_500w = save_dir + '/noDA-deep_hla_500w-T1DGC' + f'/acc_{digit}.json'
    #         cnn_100w = save_dir + '/noDA-deep_hla_100w-T1DGC' + f'/acc_{digit}.json'
    #         #transformer_500w = save_dir + '/noDA-by_input_len_500w-Pan-Asian' + f'/acc_{digit}.json'
    #         #transformer_100w = save_dir + '/noDA-by_input_len_100w-T1DGC' + f'/acc_{digit}.json'
    #         #ver2 = save_dir + '/noDA-by_input_len_70w_2-Pan-Asian' + f'/test_phased_evals_mean_{digit}.json'
    #         #cnn_500w = save_dir + '/noDA-deep_hla_500w-Pan-Asian' + f'/acc_{digit}.json'
    #         #cnn_100w = save_dir + '/noDA-deep_hla_100w-T1DGC' + f'/acc_{digit}.json'
    #         transformer_70w = save_dir + '/noDA-by_input_len_70w_1-Pan-Asian' + f'/acc_{digit}.json'
    #         storer = ComparedResultStorer(dim2_DA=dim2_DA, dim4_DA=dim4_DA, dim8_DA=dim8_DA, by_input_len_base_DA=by_input_len_base_DA, by_input_len_DA=by_input_len_DA, cnn_DA=cnn_DA, by_input_len_base=by_input_len_base, by_input_len=by_input_len, cnn=cnn)
    #         #storer = ComparedResultStorer(transformer_100w=transformer_100w, cnn_100w=cnn_100w, transformer_500w=transformer_500w, cnn_500w=cnn_500w)
    #         #####
    #         storer.store_acc_curve(save_dir, digit)
    else:
        raise

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--settings', help='path to config file', 
        default="./resources/trials/trial.yaml",
        dest='settings')
    
    parser.add_argument(
        '--phases', 
        default=[0.0, 0.005, 0.01, 0.05, 0.1],
        dest='phases'
    )

    parser.add_argument(
        '--compare',
        default=False,
        dest='compare'
    )

    parser.add_argument(
        '--exp_name',
        default='trial-T1DGC',
        dest='exp_name'
    )

    parser.add_argument(
        '--density',
        default=False,
        dest='density'
    )

    parser.add_argument(
        '--acc',
        default=False,
        dest='acc'
    )

    args = parser.parse_args()
    
    run_eval(args)