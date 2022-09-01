
from codes.eval_model import ModelEvaluator
from argparse import Namespace
from codes.supports.utils import path_to_dict

def run_eval(args: Namespace) -> None:
    """
    Args:
        args (Namespace)
    Returns:
        None
    """
    evaluator = ModelEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--settings', 
        default="./resources/settings.yaml",
        dest='settings')

    parser.add_argument(
        '--ref_name', 
        default="Pan-Asian_REF",
        dest='ref_name')

    parser.add_argument(
        '--data_dir', 
        default="Pan-Asian",
        dest='data_dir')

    args = parser.parse_args()
    settings = path_to_dict(args.settings)
    settings['ref_name'] = args.ref_name
    settings['data_dir'] = args.data_dir

    run_eval(settings)