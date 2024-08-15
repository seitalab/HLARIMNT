import os
from collections import defaultdict

import yaml
import pandas as pd

from make_fig02a import load_csv, extract_data, main

cfgfile = "./settings.yaml"
with open(cfgfile, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    
    experiment = "exp14_Equal"
    for metric in cfg["metric_dict"].keys():
        main(experiment, metric, weight_by_freq=False)
        main(experiment, metric, weight_by_freq=True)
    print("Done")