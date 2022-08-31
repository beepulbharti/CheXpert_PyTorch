import numpy as np
from sklearn.model_selection import StratifiedKFold
from balancers import BinaryBalancer

def get_splits(data,n_splits):
    x_ind = np.linspace(0,len(data)-1,len(data)).astype('int')
    a = data['Sex']
    skf = StratifiedKFold(n_splits=n_splits)
    split_indices = skf.split(x_ind,a)
    return split_indices

# Functions to calculate the TPRs and FPRs with respect to A
def calculate_bias_metrics(df):
    a = df.a.values
    y = np.array(df.y.values)
    y_ = np.array(df.y_hat.values)
    pb = BinaryBalancer(y=y,y_=y_,a=a,summary=False)
    alpha = pb.group_rates[1.0].tpr
    beta = pb.group_rates[0.0].tpr
    tau = pb.group_rates[1.0].fpr
    phi = pb.group_rates[0.0].fpr
    return alpha,beta,tau,phi

# Calculating general upper and lower bounds
def calc_gen_bounds(alpha,beta,U,r,s):
    if s*alpha + r*beta > 0.5*(s+r):
        ub = alpha - beta + U*(alpha/r + beta/s)
        lb = alpha - beta - U*(alpha/r + beta/s)
    else:
        ub = alpha - beta - U*(alpha/r + beta/s) + U*(r+s)/(r*s)
        lb = alpha - beta + U*(alpha/r + beta/s) - U*(r+s)/(r*s)
    return ub, lb

# Function to post process y_hat
def eo_postprocess(df):
    a = df.a.values
    y = np.array(df.y.values)
    y_ = np.array(df.y_prob.values)
    fair_model = BinaryBalancer(y=y,y_=y_,a=a,summary=False)
    fair_model.adjust(goal='odds', summary=False)
    fair_yh = fair_model.predict(y_,a)
    return fair_yh, fair_model

def get_base_rates(y,a,return_dict=False):
    g = np.vstack([y,a]).T
    groups = np.zeros(len(y))
    group_list = [np.array([1,1]),np.array([1,0]),np.array([0,1]),np.array([0,0])]
    for i, gr in enumerate(group_list):
        gr_ind = np.where(np.all(g == gr,axis=1))
        groups[gr_ind] = (i+1)
    r = np.sum(groups == 1)/len(y)
    s = np.sum(groups == 2)/len(y)
    v = np.sum(groups == 3)/len(y)
    w = np.sum(groups == 4)/len(y)
    rates = {'r':r,'s':s,'v':v,'w':w}
    if return_dict == True:
        return rates, groups
    else:
        return rates