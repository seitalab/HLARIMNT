import os
from collections import defaultdict

import yaml
import pandas as pd

from make_fig03a import load_csv, calc_means

cfgfile = "./settings.yaml"
with open(cfgfile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

def extract_data(
    df, 
    digit, 
    metric, 
    target_freq,
    weight_by_freq=False
):
    if digit is not None:
        df_target = df[df.digit == digit]
    else:
        df_target = df

    # Filter by target_freq
    if target_freq is not None:
        if target_freq > 0:
            df_target = df_target[df_target.freq >= target_freq]
        else:
            df_target = df_target[df_target.freq <= -target_freq]

    vals = df_target.loc[:, metric].values
    vals_non_null = vals[~pd.isnull(vals)]
    if weight_by_freq:
        freq = df_target.loc[:, "freq"].values
        freq = freq[~pd.isnull(vals)]
        vals_non_null = (vals_non_null * freq) / freq.sum()
        return vals_non_null.sum()
    return vals_non_null.mean()

def main(experiment, metric, weight_by_freq):
    metric_key = cfg["metric_dict"][metric]

    result = defaultdict(list)
    for method in cfg["methods"]:
        dataset = cfg["experiment_to_dataset"][experiment]
        for seed in cfg["seeds"]:
            df = load_csv(experiment, method, dataset, seed)
            for freq in cfg["min_freqs"]:
                vals = extract_data(
                    df, 
                    cfg["target_digit"], 
                    metric_key, 
                    freq,
                    weight_by_freq=weight_by_freq                    
                )
                result[(freq, method)].append(vals)
    result = dict(result)
    result = pd.DataFrame(result).T
    # Reorder by values in gene column with order in cfg["gene"].
    result = result.loc[cfg["min_freqs"], :]
    result.loc[:, "mean"] = result.mean(axis=1)

    # save
    save_dir = os.path.join(
        cfg["save_root"], 
        cfg["experiment_to_figname"][experiment].replace("fig3", "fig2")
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