import sklearn
import numpy as np
from dataset import Chexpert_dataset
from sklearn.model_selection import StratifiedKFold

def get_splits(data,n_splits):
    x_ind = np.linspace(0,len(data)-1,len(data)).astype('int')
    a = data.group
    skf = StratifiedKFold(n_splits=n_splits)
    split_indices = skf.split(x_ind,a)
    return split_indices
