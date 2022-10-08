import os
import argparse
from collections import OrderedDict
import json
import pandas as pd


def make_samplebim(args):
    data_loc = f'codes/data/{args.data_dir}/'
    ref_bim_loc = data_loc + f'{args.ref}.bim'
    ref_bim = pd.read_table(ref_bim_loc, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
    ref_bim = ref_bim[(ref_bim['a1']=='A')|(ref_bim['a1']=='G')|(ref_bim['a1']=='C')|(ref_bim['a1']=='T')]
    ref_bim = ref_bim[~(ref_bim['id'].str.startswith('SNP')) & ~(ref_bim['id'].str.startswith('AA')) & ~(ref_bim['id'].str.startswith('HLA')) & ~(ref_bim['id'].str.startswith('INS'))]
    sample_bim = ref_bim.drop_duplicates(subset='pos',keep=False)
    sample_bim.to_csv(data_loc + f'/{args.ref}_sample.bim', sep='\t', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Make an HLA information file.')
    parser.add_argument('--ref', default='Pan-Asian_REF', 
                        help='HLA reference data (.bgl.phased or .haps, and .bim format).', dest='ref')

    parser.add_argument('--data_dir', default='Pan-Asian',  required=False,
                        help='Directory to store data.', dest='data_dir')                        

    args = parser.parse_args()

    make_samplebim(args)


if __name__ == '__main__':
    main()