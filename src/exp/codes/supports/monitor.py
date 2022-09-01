
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple, List, Dict
from codes.supports.storer import AllResultStorer
import pandas as pd

class Monitor:
    """
    Manage preds of one HLA gene. 
    """
    def __init__(self, hla, digit) -> None:
        """
        Arg:
            None:
        Returns:
            None
        """
        self.num_data = 0
        self.total_loss = 0
        self.x_record = None
        self.ytrue_record = None
        self.ypred_record = None
        self.freq_record = None
        self.allele_ids = None

        self.hla = hla
        self.digit = digit

    def _concat_array(self, record, new_data) -> np.ndarray:
        """
        Args:
            record (np.ndarray or None):
            new_data (np.ndarray):
        Returns:
            concat_data (np.ndarray):
        """
        if record is None:
            return new_data

        else:
            return np.concatenate([record, new_data])

    def store_loss(self, loss: float):
        """
        Args:
            loss (float) : Mini batch loss
        Returns:
            None
        """
        self.total_loss += loss

    def store_num_data(self, num_data: int):
        """
        Args:
            num_data (int): Number of data in mini batch
        Returns:
            None
        """
        self.num_data += num_data

    def store_result(self, 
                    x, 
                    y_pred, 
                    y_true
                    ) -> None:
        """
        Args:
            x (np.array) : Array of encoded SNP array in mini batch
            y_true (np.ndarray) : Array of label in mini batch
            y_pred (np.ndarray) : Array of prediction distribution
        Returns:
            None
        """
        x = x.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        
        self.x_record = self._concat_array(self.x_record, x)
        self.ytrue_record = self._concat_array(self.ytrue_record, y_true)
        self.ypred_record = self._concat_array(self.ypred_record, y_pred)
        assert len(self.ytrue_record) == len(self.ypred_record)

    def accuracy(self) -> float:
        """
        Arg:
            None
        Returns:
            score (float) : accuracy of imputation of the locus.
        """
        y_pred = np.argmax(np.exp(self.ypred_record), axis=1)
        score = accuracy_score(self.ytrue_record, y_pred)
        return score

    def _sensitivity(self, allele_id: int):
        """
        Args:
            allele_id (int) : Number that represents the allele.
        Returns:
            sensitivity (float) : sensitivity of the allele.
            skip : whether to skip that allele.
        """
        skip = False
        idx_j = np.where(self.ytrue_record == allele_id)
        m = idx_j[0].shape[0]
        if m == 0:
            skip = True
            return 0.0, skip
        D_j = np.sum((np.argmax(self.ypred_record[idx_j[0]], axis=1)==allele_id).astype(int).astype(float))
        sensitivity = D_j / m
        return sensitivity, skip

    def _ppv(self, allele_id: int):
        """
        Args:
            allele_id (int) : Number that represents the allele.
        Returns:
            ppv (float) : sensitivity of the allele.
            skip : whether to skip that allele.
        """
        skip = False
        idx_j = np.where(self.ytrue_record == allele_id)
        idx_k = np.where(self.ytrue_record != allele_id)

        D_j = np.sum((np.argmax(self.ypred_record[idx_j[0]], axis=1)==allele_id).astype(int).astype(float))  
        D_k = np.sum((np.argmax(self.ypred_record[idx_k[0]], axis=1)==allele_id).astype(int).astype(float))
        if D_j + D_k == 0:
            skip = True
            return 0.0, skip
        ppv = D_j / (D_j + D_k)
        return ppv, skip

    def _r2(self, allele_id: int):
        """
        Args:
            allele_id (int) : Number that represents the allele.
        Returns:
            r2 (float) : sensitivity of the allele.
            skip : whether to skip that allele.
        """
        ytrue_record_onehot = np.identity(self.ypred_record.shape[1])[list(self.ytrue_record.astype(int))]
        skip = False
        pred = pd.Series(np.exp(self.ypred_record)[:, allele_id])
        label = pd.Series(ytrue_record_onehot[:, allele_id])
        if np.all(label == 0.0) or np.all(label == 1.0):
            skip = True
            return 0.0, skip
        r2= pred.corr(label)
        return r2, skip

    def _probability(self, id_num, allele_id):
        """
        Args:
            id_num (int) : Numbers of alleles the gene has.
            allele_id (int) : Number that represents the allele.
        Returns:
            probability (float) : sensitivity of the allele.
            skip : whether to skip that allele.
        """
        onehot = np.identity(id_num)[self.ytrue_record.astype(int)]
        probability = np.sum(onehot*np.exp(self.ypred_record), 1)
        skip = False
        if np.all(onehot[:,allele_id] == 0):
            skip = True
            return 0.0, skip
        probability = np.mean(probability[allele_id==self.ytrue_record])
        return probability, skip

    def make_evals(self, hla_info) -> None:
        """
        Args:
            hla_info (Dict) : Information about HLA.
        Returns:
            result_dict (Dict) : Values of indices.
        """
        result_dict = {'hla':[], 'digit':[],'allele':[], 'r2':[], 'probability':[], 'ppv':[], 'sensitivity':[]}

        id_num = len(hla_info[self.hla][self.digit])
        for id in range(id_num):
            r2, skip_r2 = self._r2(id)
            ppv, skip_ppv = self._ppv(id)
            sensitivity, skip_sens = self._sensitivity(id)
            probability, skip_prob = self._probability(id_num, id)

            if not skip_r2:
                result_dict['r2'].append(r2)
            else:
                result_dict['r2'].append(np.nan)

            if not skip_ppv:
                result_dict['ppv'].append(ppv)
            else:
                result_dict['ppv'].append(np.nan)

            if not skip_sens:
                result_dict['sensitivity'].append(sensitivity)
            else:
                result_dict['sensitivity'].append(np.nan)

            if not skip_prob:
                result_dict['probability'].append(probability)
            else:
                result_dict['probability'].append(np.nan)  

            result_dict['hla'].append(self.hla)
            result_dict['digit'].append(self.digit)
            result_dict['allele'].append(hla_info[self.hla][self.digit][id])
        return result_dict

class EarlyStopper:

    def __init__(self, patience: int) -> None:
        """
        Manage the early stopping during the training.
        Args:
            patience (int) : Criteria of early stopping
        Returns:
            None
        """
        self.patience = patience
        self.num_bad_count = 0
        self.val_best = -np.inf
        self.train_best = -np.inf
        self.best_epoch = 1

    def show_best(self) -> Dict:
        return {'epoch':self.best_epoch, 'val_acc': self.val_best, 'train_acc': self.train_best}
    
    def stop_training(self, epoch_acc_val: float, epoch_acc_train, epoch: int) -> bool:
        """
        Args:
            epoch_acc_val (float) : val accuracy of the epoch.
            epoch_acc_train (float) : train accuracy of the epoch.
            epoch (int) : epoch.
        Returns:
            stop_train (bool) : whether stop training or not
        """
        if epoch_acc_train > self.train_best:
            self.train_best = epoch_acc_train

        if epoch_acc_val <= self.val_best:
            self.num_bad_count += 1

        else:
            self.num_bad_count = 0
            self.val_best = epoch_acc_val
            self.best_epoch = epoch

        if self.num_bad_count > self.patience:
            stop_train = True
            print('Early stop.')
        else:
            stop_train = False

        return stop_train
