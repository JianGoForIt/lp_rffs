
# coding: utf-8

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import cPickle as cp
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


# In[2]:


data_path = "../../../data/census/"


# In[3]:


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


# In[4]:


X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
n_input_feat = X_train.shape[1]
args = Args(n_fp_rff=1024, n_bit=8, 
            exact_kernel=True, reg_lambda=1e-3, 
            sigma=30.0, random_seed=1, 
            data_path=data_path, do_fp=False)


# ### fixed bits results

# In[5]:


# fixed_bits_results = {}
# for n_fp_feat_budget in [1024, 4096]:
#     plt.figure()
#     for nbit in [8, 4, 2]:
#         n_quantized_rff = int(np.floor(n_fp_feat_budget / float(nbit) * 32.0) )
#         min_val = -np.sqrt(2.0/float(n_quantized_rff) )
#         max_val = np.sqrt(2.0/float(n_quantized_rff) )
#         quantizer_train = Quantizer(nbit, min_val, max_val, rand_seed=args.random_seed)
#         quantizer_test = quantizer_train
#         kernel = GaussianKernel(sigma=args.sigma)
#         kernel = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
#         kernel_mat = kernel.get_kernel_matrix(X_train, X_train, quantizer_train, quantizer_train)
#         S, U = np.linalg.eigh(kernel_mat.cpu().numpy().astype(np.float64) )
#         fixed_bits_results["budget_" + str(n_fp_feat_budget) + "fixed_" + str(nbit) ] = S
#         plt.semilogy(S, label="budget_" + str(n_fp_feat_budget) + "fixed_" + str(nbit)  )
#     plt.show()


# In[6]:


# # with open("./tmp/spectrum_fix_var_comp_fixed_rep_04_13.pkl", "w") as f:
# #     cp.dump(fixed_bits_results,f)
# with open("./tmp/spectrum_fix_var_comp_fixed_rep_04_13.pkl", "r") as f:
#     fixed_bits_results = cp.load(f)
# print fixed_bits_results.keys()


# ### variable bit spectrum

# In[7]:


def run_single_config(memory_budget, baseline_bit, average_bit_top_comp, 
                      residual_bit, percentile, reg_lambda, seed):
    data_path = "../../../data/census/"
    sigma = 30.0
    X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
    n_input_feat = X_train.shape[1]
    fp_bits = 32.0
    n_rff = memory_budget * fp_bits / float(baseline_bit)
    top_comp_budget = memory_budget - residual_bit / 32.0 * n_rff
    n_top_comp = top_comp_budget * fp_bits / float(average_bit_top_comp)
    assert residual_bit < baseline_bit
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

# In[8]:


spectrum_dict = {}
memory_budget = 1024.0


# In[9]:


# for baseline_bit in [4, 8, 2, 16]:
for seed in [1, 2, 3]:
    for baseline_bit in [2,]:
        for residual_bit in [8, 4, 2, 1]:
            for reg_lambda in [1e-6, 1e-5, 1e-4]:
                if residual_bit >= baseline_bit:
                    continue
        #         plt.figure()
        #         plt.semilogy(fixed_bits_results["budget_1024fixed_" + str(baseline_bit) ][::-1], label="fixed " + str(baseline_bit) )
                for average_bit_top_comp in [32, 16, 8, 4, 2, 1]:
        #         for average_bit_top_comp in [32, ]:
                    if average_bit_top_comp < residual_bit:
                        continue
                    for percentile in [0.0, 0.1, 1.0, 10.0]:
                        label = "mem_budget_" + str(int(memory_budget) )                               + "_baseline_bit_" + str(int(baseline_bit) )                               + "_average_bit_top_comp_" + str(int(average_bit_top_comp) )                               + "_residual_bit_" + str(int(residual_bit) )                               + "_percentile_" + str(percentile)                               + "_lambda_" + str(reg_lambda)                               + "_seed_" + str(seed)
                        print label
                        spectrum = run_single_config(memory_budget=memory_budget, 
                                             baseline_bit=baseline_bit, 
                                             average_bit_top_comp=average_bit_top_comp, 
                                             residual_bit=residual_bit, percentile=percentile, 
                                             reg_lambda=reg_lambda, seed=seed)
        #                 print spectrum                
                        spectrum_dict[label] = spectrum
        #                 plt.semilogy(spectrum, label=label)
        #         plt.grid()
        #         plt.ylim([1e-7, 1e3] )
        #         plt.legend(framealpha=0.3)
        #         plt.show()
                with open("/dfs/scratch0/zjian/lp_kernel/census_results_64_bit_fix_var_comp/fix_var_comp_l2_1024_memory_budget_0416_part2.1.pkl", "w") as f:
                    cp.dump(spectrum_dict, f)


# In[ ]:


with open("/dfs/scratch0/zjian/lp_kernel/census_results_64_bit_fix_var_comp/fix_var_comp_l2_1024_memory_budget_0416_part2.1.pkl", "w") as f:
    cp.dump(spectrum_dict, f)

