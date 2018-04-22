
# coding: utf-8

# In[16]:


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import cPickle as cp
import math
import sys, os
sys.path.append("../../utils")
sys.path.append("../")

from data_loader import load_census_data, load_census_data_part
from plot_utils import get_colors
import rff
from rff import GaussianKernel, RFF
from kernel_regressor import Quantizer, QuantizerAutoScale, KernelRidgeRegression
from bit_assignment import binary_search_bits_assignment
from pca_rff import PCA_RFF
from pca_rff_top_k import PCA_RFFTopK
import cPickle as cp
from misc_utils import Args


# In[17]:


top_comp_perc_list = [5]
reg_lambda_list = [1e-5, 1e-4, 1e-3]


# In[18]:


data_path = "../../../data/census/"


# In[19]:


def load_census_data_part(path):
  X_test = np.load(path + "X_ho.npy")
  X_train = np.load(path + "X_tr.npy")
  Y_test = np.load(path + "Y_ho.npy")
  Y_train = np.load(path + "Y_tr.npy")
  X_test = X_test.item()['X_ho']
  X_train = X_train.item()['X_tr']
  Y_test = Y_test.item()['Y_ho']
  Y_train = Y_train.item()['Y_tr']
#   s = np.arange(X_train.shape[0] )
#   np.random.seed(0)
#   np.random.shuffle(s)
#   X_train = X_train[s, :]
#   Y_train = Y_train[s]
#   X_train, Y_train, X_test, Y_test = \
#     X_train[:(s.size * 1 / 3), :], Y_train[:(s.size * 1 / 3)], X_test[:(s.size * 1 / 3), :], Y_test[:(s.size * 1 / 3)]

#   X_train, Y_train, X_test, Y_test = \
#     X_train[:(s.size * 1 / 3), :], Y_train[:(s.size * 1 / 3)], X_test[:(s.size * 1 / 3), :], Y_test[:(s.size * 1 / 3)]
  return X_train, X_test, Y_train, Y_test


# In[20]:


X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
n_input_feat = X_train.shape[1]
args = Args(n_fp_rff=1024, n_bit=8, 
            exact_kernel=True, reg_lambda=1e-3, 
            sigma=30.0, random_seed=1, 
            data_path=data_path, do_fp=False)


# ### fixed bits results

# In[21]:


# # with open("./tmp/spectrum_fix_var_comp_fixed_rep_04_13.pkl", "w") as f:
# #     cp.dump(fixed_bits_results,f)
# with open("./tmp/spectrum_fix_var_comp_fixed_rep_04_13.pkl", "r") as f:
#     fixed_bits_results = cp.load(f)
# print fixed_bits_results.keys()


# ### variable bit spectrum

# In[22]:


def run_single_config(memory_budget, top_component_perc, average_bit_top_comp, 
                      residual_bit, percentile, reg_lambda, seed):
    # if top component perc is 1, it means we take 1% components to use
    data_path = "../../../data/census/"
    sigma = 30.0
    X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
    n_input_feat = X_train.shape[1]
    fp_bits = 32.0
    n_rff = math.floor(memory_budget * fp_bits / (top_component_perc / float(100) * average_bit_top_comp + residual_bit) )
    top_comp_budget = memory_budget - residual_bit / 32.0 * n_rff
    n_top_comp = int(math.floor(n_rff * top_component_perc / float(100) ) )    
    assert np.abs(n_top_comp - np.floor(n_top_comp) ) < 1e-6
    assert np.abs(n_top_comp - np.ceil(n_top_comp) ) < 1e-6
    assert np.abs(n_rff - np.floor(n_rff) ) < 1e-6
    assert np.abs(n_rff - np.ceil(n_rff) ) < 1e-6
    quantizer = lambda nbit, min_val, max_val, rand_seed:           QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=rand_seed, percentile=percentile)
    # get a lp precision PCA RFF topK
    kernel = GaussianKernel(sigma=sigma)
    kernel = PCA_RFFTopK(int(n_rff), n_input_feat, kernel, rand_seed=seed)
    kernel.setup(X_train, n_top_comp=int(n_top_comp), n_fp_feat_budget=memory_budget, residual_bit=residual_bit, bits_upperbound=32)
    regressor = KernelRidgeRegression(kernel, reg_lambda=reg_lambda)
    regressor.fit(X_train, Y_train, quantizer=quantizer)
    train_error = regressor.get_train_error()
    pred = regressor.predict(X_test, quantizer_train=quantizer, quantizer_test=quantizer)
    test_error = regressor.get_test_error(Y_test)
    
    print "train test l2 ", train_error, test_error, reg_lambda, seed
#     kernel_mat = kernel.get_kernel_matrix(X_train, X_train, quantizer1=quantizer, quantizer2=quantizer)
#     S, U = np.linalg.eigh(kernel_mat.cpu().numpy().astype(np.float64) )
    return (float(train_error), float(test_error) )


# # Memory budget 1024

# In[23]:


# with open("./tmp/fix_var_comp_l2_1024_memory_budget_0416_part1.1.pkl", "r") as f:
#     spectrum_dict = cp.load(f)
spectrum_dict = {}
memory_budget = 1024.0


# In[24]:


# print spectrum_dict.keys()


# In[25]:


# for baseline_bit in [4, 8, 2, 16]:
cnt = 0
for seed in [1, 2, 3]:
#     for top_comp_perc in [5, 1, 10]:
    for top_comp_perc in top_comp_perc_list:
        for residual_bit in [8, 4, 2, 1]:
            for reg_lambda in reg_lambda_list:
#             for reg_lambda in [1e-2, 1e1, 1e0]:
                for average_bit_top_comp in [32, 16, 8, 4, 2, 1]:
                    if average_bit_top_comp < residual_bit:
                        continue
                    for percentile in [0.0, 0.1]:
                        label = "mem_budget_" + str(int(memory_budget) )                               + "_top_comp_perc_" + str(top_comp_perc)                               + "_average_bit_top_comp_" + str(int(average_bit_top_comp) )                               + "_residual_bit_" + str(int(residual_bit) )                               + "_percentile_" + str(percentile)                               + "_lambda_" + str(reg_lambda)                               + "_seed_" + str(seed)
                        print label
#                         if label in spectrum_dict.keys():
#                             continue
                        spectrum = run_single_config(memory_budget=memory_budget, 
                                             top_component_perc=top_comp_perc, 
                                             average_bit_top_comp=average_bit_top_comp, 
                                             residual_bit=residual_bit, percentile=percentile, 
                                             reg_lambda=reg_lambda, seed=seed)
                        spectrum_dict[label] = spectrum
                        cnt += 1
                    with open("/dfs/scratch0/zjian/lp_kernel/census_results_64_bit_fix_var_comp_top_comp_perc/fix_var_comp_l2_1024_memory_budget_0421_part1.1.pkl", "w") as f:
                        cp.dump(spectrum_dict, f)


# In[ ]:


# with open("/dfs/scratch0/zjian/lp_kernel/census_results_64_bit_fix_var_comp_top_comp_perc/fix_var_comp_l2_1024_memory_budget_0421_part1.1.pkl", "w") as f:
#     cp.dump(spectrum_dict, f)

