import os
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd

from make_fig03a import load_csv, extract_data, calc_means

cfgfile = "./settings.yaml"
with open(cfgfile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

def main(experiment, metric, weight_by_freq):
    metric_key = cfg["metric_dict"][metric]

    result = defaultdict(list)
    for method in cfg["methods"]:
        dataset = cfg["experiment_to_dataset"][experiment]
        for seed in cfg["seeds"]:
            df = load_csv(experiment, method, dataset, seed)
            for gene in cfg["genes"]:
                vals = extract_data(
                    df, 
                    gene, 
                    cfg["target_digit"], 
                    metric_key, 
                    max_freq=cfg["infreq_threshold"], 
                    weight_by_freq=weight_by_freq
                )
                result[(gene, method)].append(vals)
    result = dict(result)
    result = pd.DataFrame(result).T
    
    # Reorder by values in gene column with order in cfg["gene"].
    result = result.loc[cfg["genes"], :]
    
    result = calc_means(result)

    # save
    save_dir = os.path.join(
        cfg["save_root"], 
        cfg["experiment_to_figname"][experiment] + "_infreq"
    )
    os.makedirs(save_dir, exist_ok=True)
    metric_idx = cfg["metric_idx"][metric]
    save_path = os.path.join(save_dir, f"{metric_idx:02d}_{metric}.csv")
    if weight_by_freq:
        save_path = save_path.replace(".csv", "_weighted.csv")
    result.to_csv(save_path)

if __name__ == "__main__":
    
    experiment = "exp14_Pan-Asian"
    for metric in cfg["metric_dict"].keys():
        main(experiment, metric, weight_by_freq=False)
        main(experiment, metric, weight_by_freq=True)
    print("Done")