import os
from collections import defaultdict

import yaml
import pandas as pd

from make_fig03a import load_csv, calc_means
from make_fig02a import extract_data

cfgfile = "./settings.yaml"
with open(cfgfile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

def main(metric, weight_by_freq, infreq_only=False):
    digit = "4-digit"
    metric_key = cfg["metric_dict"][metric]

    if infreq_only:
        target_freq = cfg["infreq_threshold"]
    else:
        target_freq = None

    result = defaultdict(list)
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            for experiment, datasize in cfg["datasizes"].items():
                dataset = cfg["experiment_to_dataset"][experiment]
                df = load_csv(experiment, method, dataset, seed)
                vals = extract_data(
                    df, 
                    digit, 
                    metric_key, 
                    target_freq=target_freq,
                    weight_by_freq=weight_by_freq
                )
                result[(experiment, method)].append(vals)
    result = dict(result)
    result = pd.DataFrame(result).T
    # Reorder by values in gene column with order in cfg["gene"].
    result = result.loc[cfg["datasizes"], :]
    result.loc[:, "mean"] = result.mean(axis=1)

    # save
    save_dir = os.path.join(
        cfg["save_root"], 
        "fig4"
    )
    if infreq_only:
        save_dir = save_dir.replace("fig4", "fig4_infreq")
    os.makedirs(save_dir, exist_ok=True)
    metric_idx = cfg["metric_idx"][metric]
    save_path = os.path.join(save_dir, f"{metric_idx:02d}_{metric}.csv")
    if weight_by_freq:
        save_path = save_path.replace(".csv", "_weighted.csv")
    result.to_csv(save_path)

if __name__ == "__main__":
    
    for metric in cfg["metric_dict"].keys():
        main(metric, weight_by_freq=False)
        main(metric, weight_by_freq=True)
        main(metric, weight_by_freq=False, infreq_only=True)
        main(metric, weight_by_freq=True, infreq_only=True)
    print("Done")