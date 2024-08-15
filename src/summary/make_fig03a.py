import os
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd

cfgfile = "./settings.yaml"
with open(cfgfile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

def load_csv(experiment, method, dataset, seed):
    filename = os.path.join(
        cfg["root_path"],
        experiment,
        f"{method}-{dataset}",
        f"seed{seed}/csvs",
        cfg["result_file"]
    )
    df = pd.read_csv(filename, index_col=0)
    return df

def extract_data(
    df, 
    hla, 
    digit, 
    metric, 
    max_freq=None, 
    weight_by_freq=False
):
    if digit is not None:
        target_row = (df.hla == hla) & (df.digit == digit)
    else:
        target_row = (df.hla == hla)
    df_target = df[target_row]

    # Filter by max_freq
    if max_freq is not None:
        df_target = df_target[df_target.freq <= max_freq]

    vals = df_target.loc[:, metric].values
    vals_non_null = vals[~pd.isnull(vals)]
    if pd.isnull(vals).all():
        return np.nan
    if weight_by_freq:
        freq = df_target.loc[:, "freq"].values
        freq = freq[~pd.isnull(vals)]
        vals_non_null = (vals_non_null * freq) / freq.sum()
        return vals_non_null.sum()
    # return vals_non_null.mean()
    return np.nanmean(vals_non_null)

def calc_means(result):
    # swap index level 0 and 1
    result = result.swaplevel(0, 1)

    # calc mean grouped by value of second index and add it to the result
    result_mean = result.groupby(level=0).mean()
    
    # add result_mean to the result, with the name "mean"
    result_mean.index = pd.MultiIndex.from_product([["mean"], result_mean.index])
    result = result.swaplevel(0, 1)
    result = result.append(result_mean)

    # calc mean for each row and add it to the result
    result.loc[:, "mean"] = result.mean(axis=1)
    return result

def main(experiment, metric, weight_by_freq):
    metric_key = cfg["metric_dict"][metric]

    result = defaultdict(list)
    for method in cfg["methods"]:
        dataset = cfg["experiment_to_dataset"][experiment]
        for seed in cfg["seeds"]:
            df = load_csv(experiment, method, dataset, seed)
            for gene in cfg["genes"]:
                vals = extract_data(
                    df, gene, cfg["target_digit"], metric_key, weight_by_freq=weight_by_freq)
                result[(gene, method)].append(vals)
    result = dict(result)
    result = pd.DataFrame(result).T
    # Reorder by values in gene column with order in cfg["gene"].
    result = result.loc[cfg["genes"], :]
    
    result = calc_means(result)
    
    # save
    save_dir = os.path.join(
        cfg["save_root"], cfg["experiment_to_figname"][experiment])
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