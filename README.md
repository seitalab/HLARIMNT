# HLARIMNT
Codes for training and evaluation, used in "Efficient HLA imputation from sequential SNPs data by Transformer".

(Part of our code is adopted from [A deep learning method for HLA imputation and
trans-ethnic MHC fine-mapping of type 1 diabetes](https://github.com/tatsuhikonaito/DEEP-HLA).)

## Proposal Part
The structure of the Transformer-based model we proposed in this study is stored in the following directory.  

***src/exp/codes/architectures/***

In addition, detailed settings such as hyperparameters are described in the following configuration file.

***src/exp/resources/settings.yaml***

## Data preparation
Pan-Asian reference panel is available [here](http://software.broadinstitute.org/mpg/snp2hla/) and T1DGC reference panel is available [here](https://repository.niddk.nih.gov/studies/t1dgc/) after the registration process.

Put the directory that contains the reference panels in ***src/exp/codes/data/*** .

Then, if your directory is named "Pan-Asian", there will be a path ***src/exp/codes/data/Pan-Asian/Pan-Asian_REF.bim***,etc.

## Experiment
- Run the following command to move to the directory for experiment.
```
cd src/exp
```
- Run the following command to create information file of the reference panel.
```
python make_hlainfo.py --ref REFERENCE_PANEL_NAME --data_dir DATA_DIR_NAME
```
ex) If reference panels are "Pan-Asian_REF.bim/bgl.phased" and they are stored in "Pan-Asian" directory, REFERENCE_PANEL_NAME will be "Pan-Asian_REF" and DATA_DIR_NAME will be "Pan-Asian".

- Run the following command to create sample.bim, which contains only SNP information in the .bim file in reference panels.
```
python make_samplebim.py --ref REFERENCE_PANEL_NAME --data_dir DATA_DIR_NAME
```

- Run the following command to train the model in partitioned data.
```
python run_train.py --ref REFERENCE_PANEL_NAME --data_dir DATA_DIR_NAME
```
This command will make a directory ***src/exp/REFERENCE_PANEL_NAME***, which contains "outputs" directory and "processed_data" directory. The former one contains models and logs generated during the training in each fold and the latter one contains encoded data and data loaders for each fold.

If you don't want to perform a cross-validation (just want to train the model), you should set the variant "fold_num" in ***src/exp/resources/settings.yaml*** to 1 (default is 5).

- Run the following command to calculate accuracy for each allele in test data of each fold.
```
python run_eval.py --ref REFERENCE_PANEL_NAME --data_dir DATA_DIR_NAME
```
This command will make a csv file ***outputs/FOLD_NUM/results/test_evals.csv*** for each fold (FOLD_NUM).