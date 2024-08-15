
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
            freq (np.ndarray) : Array of allele frequency in mini batch
            allele_id (np.ndarray) : Array of allele id
        Returns:
            None
        """
        x = x.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        
        self.x_record = self._concat_array(self.x_record, x)
        self.ytrue_record = self._concat_array(self.ytrue_record, y_true)
        self.ypred_record = self._concat_array(self.ypred_record, y_pred)
        #self.freq_record = self._concat_array(self.freq_record, y_freq)
        assert len(self.ytrue_record) == len(self.ypred_record)

    def accuracy(self) -> float:
        """
        Arg:
            None
        Returns:
            score (float) : accuracy of the HLA gene.
        """
        y_pred = np.argmax(np.exp(self.ypred_record), axis=1)
        score = accuracy_score(self.ytrue_record, y_pred)
        
        return score

    def _ppv(self, allele_id: int) -> Tuple[float, bool]:
        """
        Calculate PPV of one allele
        Arg:
            allele_id (int)
        Returns:
            ppv_of_allele
            skip
        """
        skip = False
        idx_j = np.where(self.ytrue_record == allele_id)
        idx_k = np.where(self.ytrue_record != allele_id)
        m = idx_j[0].shape[0]

        if m == 0:
            skip = True
            return 0.0, skip
        D_j = np.sum(np.exp(self.ypred_record[idx_j[0]][:, int(allele_id)]))
        D_k = np.sum(np.exp(self.ypred_record[idx_k[0]][:, int(allele_id)]))
        if D_j+D_k == 0.0:
            return 0.0, skip
        ppv_of_allele = D_j / (D_k + D_j)
        return ppv_of_allele, skip

    def _sensitivity(self, allele_id: int) -> Tuple[float, bool]:
        """
        Arg:
            allele_id (int)
        Returns:
            sens_of_allele
            skip
        """        
        skip = False
        idx_j = np.where(self.ytrue_record == allele_id)
        m = idx_j[0].shape[0]
        if m == 0:
            skip = True
            return 0.0, skip
        D_j = np.sum(np.exp(self.ypred_record[idx_j[0]][:, int(allele_id)]))
        sens_of_allele = D_j / m
        return sens_of_allele, skip

    def _sensitivity_revise(self, allele_id):
        skip = False
        idx_j = np.where(self.ytrue_record == allele_id)
        m = idx_j[0].shape[0]
        if m == 0:
            skip = True
            return 0.0, skip
        #onehot = np.identity(id_num)[self.ytrue_record[idx_j[0]].astype(int)]
        D_j = np.sum((np.argmax(self.ypred_record[idx_j[0]], axis=1)==allele_id).astype(int).astype(float))
        return D_j / m, skip

    def _ppv_revise(self, allele_id):
        skip = False
        idx_j = np.where(self.ytrue_record == allele_id)
        idx_k = np.where(self.ytrue_record != allele_id)
        #onehot = np.identity(id_num)[self.ytrue_record[idx_j[0]].astype(int)]
        D_j = np.sum((np.argmax(self.ypred_record[idx_j[0]], axis=1)==allele_id).astype(int).astype(float))  
        D_k = np.sum((np.argmax(self.ypred_record[idx_k[0]], axis=1)==allele_id).astype(int).astype(float))
        if D_j + D_k == 0:
            skip = True
            return 0.0, skip
        return D_j / (D_j + D_k), skip

    def _fscore(self, ppv: float, sens: float) -> float:
        """
        Arg:
            None
        Returns:
            sens_list (np.ndarray) : sensitivity of each phase of freq.
        """        
        return 2 * ppv * sens / (ppv + sens)

    def _r2(self, allele_id):
        ytrue_record_onehot = np.identity(self.ypred_record.shape[1])[list(self.ytrue_record.astype(int))]
        skip = False
        pred = pd.Series(np.exp(self.ypred_record)[:, allele_id])
        label = pd.Series(ytrue_record_onehot[:, allele_id])
        if np.all(label == 0.0) or np.all(label == 1.0):
            skip = True
            return 0.0, skip
        r2= pred.corr(label)
        return r2, skip

    def _r2_01(self, allele_id):
        ytrue_record_onehot = np.identity(self.ypred_record.shape[1])[list(self.ytrue_record.astype(int))]
        skip = False
        ypred_onehot = np.identity(self.ypred_record.shape[1])[np.argmax(np.exp(self.ypred_record),axis=1)]
        pred = pd.Series(ypred_onehot[:, allele_id])
        label = pd.Series(ytrue_record_onehot[:, allele_id])
        if np.all(label == 0.0) or np.all(label == 1.0):
            skip = True
            return 0.0, skip
        r2= pred.corr(label)
        return r2, skip    

    def _conc(self, allele_id):
        ytrue_record_onehot = np.identity(self.ypred_record.shape[1])[list(self.ytrue_record.astype(int))]
        skip = False
        one_hot = pd.Series(ytrue_record_onehot[:, allele_id])
        if np.all(one_hot == 0):
            skip = True
            return 0.0, skip
        preds = np.exp(self.ypred_record)[ytrue_record_onehot[:,allele_id]==1.0]
        label = np.zeros(self.ypred_record.shape[1])
        label[allele_id] = 1.0
        #print(label)
        label = pd.Series(label)
        conc = []
        for pred in preds:
            conc_tmp = pd.Series(pred).corr(label)
            conc.append(conc_tmp)
        conc= np.mean(conc)
        return conc, skip

    def _R2(self, id_num):
        onehot = np.identity(id_num)[self.ytrue_record.astype(int)]
        #print(np.mean(onehot, axis=1))
        onehot_diff = onehot - np.mean(onehot, axis=1).reshape(onehot.shape[0], -1)
        pred_diff = self.ypred_record - np.mean(self.ypred_record, axis=1).reshape(onehot.shape[0], -1)
        R2 = np.sum(onehot_diff*pred_diff, 1) / (np.sqrt(np.sum(onehot_diff ** 2, 1)) * np.sqrt(np.sum(pred_diff ** 2, 1)))
        return R2

    def _correct(self, id_num):
        onehot = np.identity(id_num)[self.ytrue_record.astype(int)]
        correct = (np.argmax(self.ypred_record, axis=1)==np.argmax(onehot, axis=1)).astype(int).astype(float)
        return correct

    def _confidence(self, id_num, allele_id):
        onehot = np.identity(id_num)[self.ytrue_record.astype(int)]
        confidence = np.sum(onehot*np.exp(self.ypred_record), 1)
        skip = False
        if np.all(onehot[:,allele_id] == 0):
            skip = True
            return 0.0, skip
        return np.mean(confidence[allele_id==self.ytrue_record]), skip

    def make_evals_by_freq(self, id_num: int, freqs: List) -> None:
        """
        Args:
            id_num (int) : # of kinds of alleles in the HLA gene of the digit.
            freqs (List) : List of freq of each allele.
        Returns:
            None
        """
        result_dict = {'hla':[], 'digit':[],'r2':[], 'r2_01':[], 'concordance':[], 'confidence':[], 'ppv':[], 'sens':[], 'fscore':[], 'freq':[]}
        sample_result_dict = {}
        #sample_result_dict = {'hla':[], 'digit':[], 'R2':[], 'correct':[], 'freq':[]}
        sample_result_dict['R2'] = list(self._R2(id_num))
        #sample_result_dict['confidence'] = list(self._confidence(id_num))
        sample_result_dict['accuracy'] = list(self._correct(id_num))
        sample_result_dict['freq'] = [freqs[int(id)] for id in self.ytrue_record]
        sample_result_dict['hla'] = [self.hla for i in range(len(sample_result_dict['freq']))]
        sample_result_dict['digit'] = [self.digit for i in range(len(sample_result_dict['freq']))]

        for id in range(id_num):
            r2, skip_r2 = self._r2(id)
            r2_01, skip_r2_01 = self._r2_01(id)
            conc, skip_conc = self._conc(id)
            ppv, skip_ppv = self._ppv_revise(id)
            sens, skip_sens = self._sensitivity_revise(id) #self._sensitivity(id)
            confidence, skip_conf = self._confidence(id_num, id)
            if not skip_ppv and not skip_sens :
                if ppv != 0.0 and sens != 0.0:
                    fscore = self._fscore(ppv, sens)
                else:
                    fscore = 0.0
            else:
                fscore = np.nan

            freq = freqs[id]
            result_dict['hla'].append(self.hla)
            result_dict['digit'].append(self.digit)
            if not skip_r2:
                result_dict['r2'].append(r2)
            else:
                result_dict['r2'].append(np.nan)
            if not skip_r2_01:
                result_dict['r2_01'].append(r2_01)
            else:
                result_dict['r2_01'].append(np.nan)
            if not skip_conc:
                result_dict['concordance'].append(conc)
            else:
                result_dict['concordance'].append(np.nan)
            if not skip_ppv:
                result_dict['ppv'].append(ppv)
            else:
                result_dict['ppv'].append(np.nan)
            if not skip_sens:
                result_dict['sens'].append(sens)
            else:
                result_dict['sens'].append(np.nan)
            if not skip_conf:
                result_dict['confidence'].append(confidence)
            else:
                result_dict['confidence'].append(np.nan)            
            result_dict['fscore'].append(fscore)
            result_dict['freq'].append(freq)
        return result_dict, sample_result_dict

class EarlyStopper:

    def __init__(self, patience: int) -> None:
        """
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
            epoch_acc (float) : val accuracy of the epoch
            epoch (int)
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
