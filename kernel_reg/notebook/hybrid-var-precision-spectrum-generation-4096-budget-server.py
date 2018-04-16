
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
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

# #   X_train, Y_train, X_test, Y_test = \
# #     X_train[:(s.size * 1 / 3), :], Y_train[:(s.size * 1 / 3)], X_test[:(s.size * 1 / 3), :], Y_test[:(s.size * 1 / 3)]
  return X_train, X_test, Y_train, Y_test


# In[4]:


X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
n_input_feat = X_train.shape[1]
args = Args(n_fp_rff=1024, n_bit=8, 
            exact_kernel=True, reg_lambda=1e-3, 
            sigma=30.0, random_seed=3, 
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


# with open("./tmp/spectrum_fix_var_comp_fixed_rep_04_13.pkl", "w") as f:
#     cp.dump(fixed_bits_results,f)
with open("./tmp/spectrum_fix_var_comp_fixed_rep_04_13.pkl", "r") as f:
    fixed_bits_results = cp.load(f)
print fixed_bits_results.keys()


# ### variable bit spectrum

# In[7]:


def run_single_config(memory_budget, baseline_bit, average_bit_top_comp, 
                      residual_bit, percentile):
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
    
    print n_rff, X_train.shape
    
    kernel = PCA_RFFTopK(int(n_rff), n_input_feat, kernel, rand_seed=2)
    kernel.setup(X_train, n_top_comp=int(n_top_comp), n_fp_feat_budget=memory_budget, residual_bit=residual_bit, bits_upperbound=32)
    kernel_mat = kernel.get_kernel_matrix(X_train, X_train, quantizer1=quantizer, quantizer2=quantizer)
    S, U = np.linalg.eigh(kernel_mat.cpu().numpy().astype(np.float64) )
    return S[::-1]


# # Memory budget 4096

# In[8]:


spectrum_dict = {}
memory_budget = 4096.0


# In[9]:


for baseline_bit in [4, 8, 2, 16]:
    for residual_bit in [8, 4, 2, 1]:
        if residual_bit >= baseline_bit:
            continue
#         plt.figure()
        plt.semilogy(fixed_bits_results["budget_1024fixed_" + str(baseline_bit) ], label="fixed " + str(baseline_bit) )
        for average_bit_top_comp in [32, 16, 8, 4, 2, 1]:
#         for average_bit_top_comp in [32, ]:
            if average_bit_top_comp < residual_bit:
                continue
            for percentile in [0.0, 0.1, 1.0, 10.0]:
                label = "mem_budget_" + str(int(memory_budget) )                       + "_baseline_bit_" + str(int(baseline_bit) )                       + "_average_bit_top_comp_" + str(int(average_bit_top_comp) )                       + "_percentile_" + str(percentile)
                print label
                spectrum = run_single_config(memory_budget=memory_budget, 
                                     baseline_bit=baseline_bit, 
                                     average_bit_top_comp=average_bit_top_comp, 
                                     residual_bit=residual_bit, percentile=percentile)
#                 print spectrum                
                spectrum_dict[label] = spectrum.copy()
                exit(0)
#                 plt.semilogy(spectrum, label=label)
#         plt.grid()
#         plt.ylim([1e-7, 1e3] )
#         plt.legend(framealpha=0.3)
#         plt.show()
#         with open("./tmp/fix_var_comp_spectrum_4096_memory_budget_0415.pkl", "w") as f:
#             cp.dump(spectrum_dict, f)


# In[ ]:


with open("./tmp/fix_var_comp_spectrum_1024_memory_budget_0415.pkl", "w") as f:
    cp.dump(spectrum_dict, f)


# In[12]:


spectrum_dict.keys()


# In[50]:


# test
plt.figure()
percentile0 = run_single_config(memory_budget=1024, baseline_bit=8, average_bit_top_comp=16, 
                      residual_bit=4, percentile=0.0)
percentile01 = run_single_config(memory_budget=1024, baseline_bit=8, average_bit_top_comp=16, 
                      residual_bit=4, percentile=0.1)
percentile1 = run_single_config(memory_budget=1024, baseline_bit=8, average_bit_top_comp=16, 
                      residual_bit=4, percentile=1.0)
plt.semilogy(percentile0, label="perc 0")
plt.semilogy(percentile01, label="perc 0.1")
plt.semilogy(percentile1, label="perc 1.0")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


def run_var_precision(memory_budget, baseline_bits, residual_bits, top_k_ave_bits):
    n_baseline_feat = memory_budget / baseline_bits
    component_budget 


# In[ ]:


def plot_with_percentile(upper, lower):
    spectrum_dict = {}
    plt.figure()
#     for n_top_comp in [0, ]:
    for n_top_comp in [0, 10, 100, 1000]:
        print "percent ", n_top_comp / float(rff.shape[0] )
        rff_offset = np.mean(rff, axis=0)
        rff_center = rff - rff_offset
        main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
        residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
#         dy_range = np.max(residual, axis=0) - np.min(residual, axis=0)
#         dy_range_1 = np.percentile(residual, q=99, axis=0) - np.percentile(residual, q=1, axis=0)
#         dy_range_10 = np.percentile(residual, q=90, axis=0) - np.percentile(residual, q=10, axis=0)

        kernel_approx_error_list = []
        for nbit in [1, 2, 4, 8, 16, 32]:
            min_val = np.percentile(residual, q=lower, axis=0)
            max_val = np.percentile(residual, q=upper, axis=0)
            residual_clamp = np.clip(residual, min_val, max_val)
#             print np.max(residual_clamp, axis=0), max_val
#             print np.min(residual_clamp, axis=0), min_val
            min_val = np.tile(np.percentile(residual, q=lower, axis=0).reshape(1, min_val.size), (rff.shape[0], 1) )
            max_val = np.tile(np.percentile(residual, q=upper, axis=0).reshape(1, max_val.size), (rff.shape[0], 1) )
            quantizer = Quantizer(nbit, torch.DoubleTensor(min_val), 
                                  torch.DoubleTensor(max_val), rand_seed=1)
            quant_residual = quantizer.quantize(torch.DoubleTensor(residual_clamp), verbose=False).cpu().numpy()
    #         quant_residual = residual
            recover_rff = np.dot(main_comp, U[:, 0:n_top_comp].T)
            recover_rff += quant_residual
            recover_rff += rff_offset        
            recover_kernel_mat = np.dot(recover_rff, recover_rff.T)
            kernel_approx_error = np.sum( (kernel_baseline.cpu().numpy() - recover_kernel_mat)**2)
            kernel_approx_error_list.append(kernel_approx_error)
            print "done bits ", nbit
#             _, S, _ = np.linalg.svd(recover_kernel_mat)
#             spectrum_dict["upper_" + str(upper) + "_n_top_" + str(n_top_comp) + "_nbit_" + str(nbit) ] = S
        plt.plot(np.sqrt(np.array(kernel_approx_error_list) / F_norm_fp_rff_sqr), label="n top comp " + str(n_top_comp) )
    #         print kernel_baseline.cpu().numpy().ravel()[:20]
    #         print recover_kernel_mat.ravel()[:20]
    #         raw_input()
        print "n top comp ", n_top_comp, kernel_approx_error_list
    plt.grid()
    plt.legend()
    ax = plt.subplot(111)
    ax.set_yscale('log')
    plt.show()
    return spectrum_dict


# In[6]:


# get a low precision PCA RFF
kernel = GaussianKernel(sigma=args.sigma)
kernel = PCA_RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed, mu=10.0)
kernel.setup(X_train, n_fp_feat_budget=n_fp_feat_budget, bits_upperbound=32)


# In[7]:


rff = kernel.get_cos_feat(X_train)
rff_test = kernel.get_cos_feat(X_test)
kernel_baseline = torch.mm(rff, torch.transpose(rff, 0, 1) )


# ### fixed bits results

# In[ ]:


fixed_bits_results = {}
plt.figure()
for nbit in [32, 16, 8, 4, 2, 1]:
    n_quantized_rff = int(np.floor(args.n_fp_rff / float(nbit) * 32.0) )
    min_val = -np.sqrt(2.0/float(n_quantized_rff) )
    max_val = np.sqrt(2.0/float(n_quantized_rff) )
    quantizer_train = Quantizer(nbit, min_val, max_val, rand_seed=args.random_seed)
#     if not args.test_var_reduce:
    quantizer_test = quantizer_train
#     else:
#     quantizer_test = None
    kernel = GaussianKernel(sigma=args.sigma)
    n_input_feat = X_train.shape[1]
    kernel = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
    lamb_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    error_list = []
    for lamb in lamb_list:
        regressor = KernelRidgeRegression(kernel, reg_lambda=lamb)
        regressor.fit(X_train, Y_train, quantizer=quantizer_train)
        pred = regressor.predict(X_test, quantizer_train=quantizer_train, quantizer_test=quantizer_test)
        test_error = regressor.get_test_error(Y_test)
        error_list.append(test_error)
    print error_list
    fixed_bits_results["fixed_" + str(nbit) ] = list(error_list)
    plt.plot(lamb_list, np.sqrt(error_list), label="fixed nbit " + str(nbit) )
plt.show()


# In[14]:


with open("./tmp/fixed_bits_test_l2_full.pkl", "w") as f:
    cp.dump(fixed_bits_results, f)


# In[15]:


with open("./tmp/fixed_bits_test_l2_full.pkl", "r") as f:
    fixed_bits_results = cp.load(f)


# In[ ]:


print fixed_bits_results.keys()
plt.figure()
for key in fixed_bits_results.keys():
    plt.plot(lamb_list, np.sqrt(fixed_bits_results[key] ), label=key)
ax = plt.subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")
plt.grid()
plt.legend()
plt.show()


# ### l2 loss visualization generation

# In[14]:


def plot_with_percentile(upper, lower, rff, rff_test):
    error_dict = {}
    for top_comp_rate in [0.01, 0.1, 1.0]: 
        plt.figure()
        kernel_approx_error_list = []
        for nbit in [16, 8, 4, 2, 1]:
            n_fp_rff = int(np.floor(args.n_fp_rff * 32.0 / float( (1 - top_comp_rate) * nbit + top_comp_rate * 32.0) ) ) 
            n_top_comp = int(np.floor(top_comp_rate * n_fp_rff) )
            print "n comp ", n_top_comp, "total ", n_fp_rff
            n_input_feat = X_train.shape[1]
            kernel = GaussianKernel(sigma=args.sigma)
            kernel = PCA_RFF(n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed, mu=10.0)
            # n_fp_feat_budget is dummy, we only want the PCA components
            kernel.setup(X_train, n_fp_feat_budget=n_fp_feat_budget, bits_upperbound=32)
            rff = kernel.get_cos_feat(X_train).cpu().numpy()
            rff_test = kernel.get_cos_feat(X_test).cpu().numpy()
            U = kernel.U.cpu().numpy()
            
            rff_offset = np.mean(rff, axis=0)
            rff_center = rff - rff_offset
            main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
            residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
            
            rff_center_test = rff_test - rff_offset
            main_comp_test = np.dot(rff_center_test, U[:, 0:n_top_comp])
            residual_test = rff_center_test - np.dot(main_comp_test, U[:, 0:n_top_comp].T)
            
            min_val = np.percentile(residual, q=lower, axis=0)
            max_val = np.percentile(residual, q=upper, axis=0)
            residual_clamp = np.clip(residual, min_val, max_val)
            residual_clamp_test = np.clip(residual_test, min_val, max_val)
            
            min_val = np.tile(np.percentile(residual, q=lower, axis=0).reshape(1, min_val.size), (rff.shape[0], 1) )
            max_val = np.tile(np.percentile(residual, q=upper, axis=0).reshape(1, max_val.size), (rff.shape[0], 1) )
            quantizer = Quantizer(nbit, torch.DoubleTensor(min_val), 
                                  torch.DoubleTensor(max_val), rand_seed=1)
            quant_residual = quantizer.quantize(torch.DoubleTensor(residual_clamp), verbose=False).cpu().numpy()
    #         quant_residual = residual
            recover_rff = np.dot(main_comp, U[:, 0:n_top_comp].T)
            recover_rff += quant_residual
            recover_rff += rff_offset        
            recover_kernel_mat = np.dot(recover_rff, recover_rff.T)
            
            quantizer = Quantizer(nbit, torch.DoubleTensor(min_val[:rff_test.shape[0], :]), 
                                  torch.DoubleTensor(max_val[:rff_test.shape[0], :]), rand_seed=1)
            quant_residual_test = quantizer.quantize(torch.DoubleTensor(residual_clamp_test), verbose=False).cpu().numpy()
    #         quant_residual = residual
            recover_rff_test = np.dot(main_comp_test, U[:, 0:n_top_comp].T)
            recover_rff_test += quant_residual_test
            recover_rff_test += rff_offset 
            
            
            error_list = []
            lamb_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
            for lamb in lamb_list:
                n_sample = recover_kernel_mat.shape[0]
                alpha = np.dot(np.linalg.inv( (recover_kernel_mat + lamb * np.eye(n_sample) ) ), Y_train)
                kernel_mat_pred = np.dot(recover_rff_test, recover_rff.T)
                prediction = np.dot(kernel_mat_pred, alpha)
                error = prediction - Y_test
                error_list.append(np.mean(error**2) )
            error_dict["top_comp_rate_" + str(top_comp_rate) + "_nbit_" + str(nbit)                        + "_upper_" + str(upper) + "_lower_" + str(lower) ] = list(error_list)
#             print("setting nbit best lambda", nbit, lamb_list[np.argmin(error_list) ] )
#             print("error list ", error_list)
            plt.plot(lamb_list, np.sqrt(error_list), label="top comp rate " + str(top_comp_rate) + " nbit " + str(nbit) )
        
        plt.grid()
        plt.legend()
        ax = plt.subplot(111)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()
    return error_dict


# In[15]:


test_error_dict_100 = plot_with_percentile(upper=100, lower=0, rff=rff, rff_test=rff_test)


# In[16]:


print test_error_dict_100.keys()
with open("./tmp/hybrid_upper_100_test_l2_full.pkl", "w") as f:
    cp.dump(test_error_dict_100, f)


# In[17]:


test_error_dict_999 = plot_with_percentile(upper=99.9, lower=0.1, rff=rff, rff_test=rff_test)


# In[ ]:


print test_error_dict_999.keys()
with open("./tmp/hybrid_upper_999_test_l2_full.pkl", "w") as f:
    cp.dump(test_error_dict_999, f)


# In[ ]:


test_error_dict_99 = plot_with_percentile(upper=99.0, lower=1.0, rff=rff, rff_test=rff_test)


# In[ ]:


print test_error_dict_99.keys()
with open("./tmp/hybrid_upper_99_test_l2_full.pkl", "w") as f:
    cp.dump(test_error_dict_99, f)


# ### spectrum generation on the counterpart on server

# In[58]:


def plot_with_percentile(upper, lower):
    spectrum_dict = {}
    plt.figure()
#     for n_top_comp in [0, ]:
    for n_top_comp in [0, 10, 100, 1000]:
        print "percent ", n_top_comp / float(rff.shape[0] )
        rff_offset = np.mean(rff, axis=0)
        rff_center = rff - rff_offset
        main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
        residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
#         dy_range = np.max(residual, axis=0) - np.min(residual, axis=0)
#         dy_range_1 = np.percentile(residual, q=99, axis=0) - np.percentile(residual, q=1, axis=0)
#         dy_range_10 = np.percentile(residual, q=90, axis=0) - np.percentile(residual, q=10, axis=0)

        kernel_approx_error_list = []
        for nbit in [1, 2, 4, 8, 16, 32]:
            min_val = np.percentile(residual, q=lower, axis=0)
            max_val = np.percentile(residual, q=upper, axis=0)
            residual_clamp = np.clip(residual, min_val, max_val)
#             print np.max(residual_clamp, axis=0), max_val
#             print np.min(residual_clamp, axis=0), min_val
            min_val = np.tile(np.percentile(residual, q=lower, axis=0).reshape(1, min_val.size), (rff.shape[0], 1) )
            max_val = np.tile(np.percentile(residual, q=upper, axis=0).reshape(1, max_val.size), (rff.shape[0], 1) )
            quantizer = Quantizer(nbit, torch.DoubleTensor(min_val), 
                                  torch.DoubleTensor(max_val), rand_seed=1)
            quant_residual = quantizer.quantize(torch.DoubleTensor(residual_clamp), verbose=False).cpu().numpy()
    #         quant_residual = residual
            recover_rff = np.dot(main_comp, U[:, 0:n_top_comp].T)
            recover_rff += quant_residual
            recover_rff += rff_offset        
            recover_kernel_mat = np.dot(recover_rff, recover_rff.T)
            kernel_approx_error = np.sum( (kernel_baseline.cpu().numpy() - recover_kernel_mat)**2)
            kernel_approx_error_list.append(kernel_approx_error)
            print "done bits ", nbit
#             _, S, _ = np.linalg.svd(recover_kernel_mat)
#             spectrum_dict["upper_" + str(upper) + "_n_top_" + str(n_top_comp) + "_nbit_" + str(nbit) ] = S
        plt.plot(np.sqrt(np.array(kernel_approx_error_list) / F_norm_fp_rff_sqr), label="n top comp " + str(n_top_comp) )
    #         print kernel_baseline.cpu().numpy().ravel()[:20]
    #         print recover_kernel_mat.ravel()[:20]
    #         raw_input()
        print "n top comp ", n_top_comp, kernel_approx_error_list
    plt.grid()
    plt.legend()
    ax = plt.subplot(111)
    ax.set_yscale('log')
    plt.show()
    return spectrum_dict


# In[11]:


# val_10 = [2445749.1793917837, 90983.05670742986, 3575.4054838067823, 11.884725505977123, 0.00018462084635229321, 4.2366721601303593e-14]
# val_100 = [755943.12067424238, 39491.256157772084, 1583.0134186175358, 5.3615856278197169, 8.2026551445892452e-05, 1.8832408799369611e-14]
# val_1000 = [32.974419743918432, 3.1134650391946259, 0.10061072304108232, 0.00034411362396194939, 5.2218140425769637e-09, 1.2198138807298336e-18]


# In[ ]:


# plt.figure()
# plt.semilogy(np.sqrt(np.array(val_10) / 217285107.943), label="10")
# plt.semilogy(np.sqrt(np.array(val_100) / 217285107.943), label="100")
# plt.semilogy(np.sqrt(np.array(val_1000) / 217285107.943), label="1000")
# plt.grid()
# plt.show()


# In[59]:


spectrum_dict = plot_with_percentile(upper=100, lower=0)
# with open("./tmp/spectrums_upper_" + str(100) + ".npy", "w") as f:
#     cp.dump(spectrum_dict, f)


# In[60]:


spectrum_dict = plot_with_percentile(upper=99, lower=1)
# with open("./tmp/spectrums_upper_" + str(99) + ".npy", "w") as f:
#     cp.dump(spectrum_dict, f)


# In[61]:


spectrum_dict = plot_with_percentile(upper=90, lower=10)
# with open("./tmp/spectrums_upper_" + str(90) + ".npy", "w") as f:
#     cp.dump(spectrum_dict, f)


# In[48]:


for n_top_comp in [0, 10, 100, 1000, 10000]:
    plt.figure()
    print "percent ", n_top_comp / float(rff.shape[0] )
    rff_offset = np.mean(rff, axis=0)
    rff_center = rff - rff_offset
    main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
    residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
    dy_range = np.max(residual, axis=0) - np.min(residual, axis=0)
    dy_range_1 = np.percentile(residual, q=99, axis=0) - np.percentile(residual, q=1, axis=0)
    dy_range_10 = np.percentile(residual, q=90, axis=0) - np.percentile(residual, q=10, axis=0)
    plt.bar(np.arange(dy_range.size), dy_range)
    plt.bar(np.arange(dy_range.size), dy_range_1)
    plt.bar(np.arange(dy_range.size), dy_range_10)
    ax = plt.subplot(111)
    ax.set_yscale("log")
    plt.show()    


# ### plot spectrums

# In[62]:


with open("./tmp/spectrums_upper_" + str(100) + ".npy", "r") as f:
    spectrums_upper_100 = cp.load(f)
with open("./tmp/spectrums_upper_" + str(99) + ".npy", "r") as f:
    spectrums_upper_99 = cp.load(f)
with open("./tmp/spectrums_upper_" + str(90) + ".npy", "r") as f:
    spectrums_upper_90 = cp.load(f)
with open("./exact_kernel_spectrum.npy", "r") as f:
    spectrums_exact = np.load(f)
with open("./spectrum_rff_pca_sqr_with_l2_n_base_feat_1024.pkl", "r") as f:
    spectrums = cp.load(f)
spectrums_fp = spectrums["fp_rff_error"]
with open("./spectrum_rff_pca_sqr_with_l2_n_base_feat_8192.pkl", "r") as f:
    spectrums_fix_bit = cp.load(f)
# np.savetxt("./fp_rff_kernel_spectrum_1024.txt", spectrums["fp_rff_error"], delimiter=",")


# In[28]:


print spectrums_upper_100.keys()


# ### plot upper 100 spectrum

# In[40]:


plt.figure()
plt.semilogy(spectrums_upper_100["upper_100_n_top_10_nbit_32"], label="32")
plt.semilogy(spectrums_upper_100["upper_100_n_top_10_nbit_16"], label="16")
plt.semilogy(spectrums_upper_100["upper_100_n_top_10_nbit_8"], label="8")
plt.semilogy(spectrums_upper_100["upper_100_n_top_10_nbit_4"], label="4")
plt.semilogy(spectrums_upper_100["upper_100_n_top_10_nbit_2"], label="2")
plt.semilogy(spectrums_upper_100["upper_100_n_top_10_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[41]:


plt.figure()
plt.semilogy(spectrums_upper_100["upper_100_n_top_100_nbit_32"], label="32")
plt.semilogy(spectrums_upper_100["upper_100_n_top_100_nbit_16"], label="16")
plt.semilogy(spectrums_upper_100["upper_100_n_top_100_nbit_8"], label="8")
plt.semilogy(spectrums_upper_100["upper_100_n_top_100_nbit_4"], label="4")
plt.semilogy(spectrums_upper_100["upper_100_n_top_100_nbit_2"], label="2")
plt.semilogy(spectrums_upper_100["upper_100_n_top_100_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[42]:


plt.figure()
plt.semilogy(spectrums_upper_100["upper_100_n_top_1000_nbit_32"], label="32")
plt.semilogy(spectrums_upper_100["upper_100_n_top_1000_nbit_16"], label="16")
plt.semilogy(spectrums_upper_100["upper_100_n_top_1000_nbit_8"], label="8")
plt.semilogy(spectrums_upper_100["upper_100_n_top_1000_nbit_4"], label="4")
plt.semilogy(spectrums_upper_100["upper_100_n_top_1000_nbit_2"], label="2")
plt.semilogy(spectrums_upper_100["upper_100_n_top_1000_nbit_2"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# ### plot upper 99

# In[65]:


plt.figure()
plt.semilogy(spectrums_upper_99["upper_99_n_top_10_nbit_32"], label="32")
plt.semilogy(spectrums_upper_99["upper_99_n_top_10_nbit_16"], label="16")
plt.semilogy(spectrums_upper_99["upper_99_n_top_10_nbit_8"], label="8")
plt.semilogy(spectrums_upper_99["upper_99_n_top_10_nbit_4"], label="4")
plt.semilogy(spectrums_upper_99["upper_99_n_top_10_nbit_2"], label="2")
plt.semilogy(spectrums_upper_99["upper_99_n_top_10_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[71]:


plt.figure()
plt.semilogy(spectrums_upper_99["upper_99_n_top_100_nbit_32"], label="32")
plt.semilogy(spectrums_upper_99["upper_99_n_top_100_nbit_16"], label="16")
plt.semilogy(spectrums_upper_99["upper_99_n_top_100_nbit_8"], label="8")
plt.semilogy(spectrums_upper_99["upper_99_n_top_100_nbit_4"], label="4")
plt.semilogy(spectrums_upper_99["upper_99_n_top_100_nbit_2"], label="2")
plt.semilogy(spectrums_upper_99["upper_99_n_top_100_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[67]:


plt.figure()
plt.semilogy(spectrums_upper_99["upper_99_n_top_1000_nbit_32"], label="32")
plt.semilogy(spectrums_upper_99["upper_99_n_top_1000_nbit_16"], label="16")
plt.semilogy(spectrums_upper_99["upper_99_n_top_1000_nbit_8"], label="8")
plt.semilogy(spectrums_upper_99["upper_99_n_top_1000_nbit_4"], label="4")
plt.semilogy(spectrums_upper_99["upper_99_n_top_1000_nbit_2"], label="2")
plt.semilogy(spectrums_upper_99["upper_99_n_top_1000_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# ### plot upper 90

# In[68]:


plt.figure()
plt.semilogy(spectrums_upper_90["upper_90_n_top_10_nbit_32"], label="32")
plt.semilogy(spectrums_upper_90["upper_90_n_top_10_nbit_16"], label="16")
plt.semilogy(spectrums_upper_90["upper_90_n_top_10_nbit_8"], label="8")
plt.semilogy(spectrums_upper_90["upper_90_n_top_10_nbit_4"], label="4")
plt.semilogy(spectrums_upper_90["upper_90_n_top_10_nbit_2"], label="2")
plt.semilogy(spectrums_upper_90["upper_90_n_top_10_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[69]:


plt.figure()
plt.semilogy(spectrums_upper_90["upper_90_n_top_100_nbit_32"], label="32")
plt.semilogy(spectrums_upper_90["upper_90_n_top_100_nbit_16"], label="16")
plt.semilogy(spectrums_upper_90["upper_90_n_top_100_nbit_8"], label="8")
plt.semilogy(spectrums_upper_90["upper_90_n_top_100_nbit_4"], label="4")
plt.semilogy(spectrums_upper_90["upper_90_n_top_100_nbit_2"], label="2")
plt.semilogy(spectrums_upper_90["upper_90_n_top_100_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[70]:


plt.figure()
plt.semilogy(spectrums_upper_90["upper_90_n_top_1000_nbit_32"], label="32")
plt.semilogy(spectrums_upper_90["upper_90_n_top_1000_nbit_16"], label="16")
plt.semilogy(spectrums_upper_90["upper_90_n_top_1000_nbit_8"], label="8")
plt.semilogy(spectrums_upper_90["upper_90_n_top_1000_nbit_4"], label="4")
plt.semilogy(spectrums_upper_90["upper_90_n_top_1000_nbit_2"], label="2")
plt.semilogy(spectrums_upper_90["upper_90_n_top_1000_nbit_1"], label="1")
plt.semilogy(spectrums_fp, label="fp")
plt.semilogy(spectrums_exact, label="exact")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_8_seed_1"], '-.', label="8 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_4_seed_1"], '-.', label="4 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_2_seed_1"], '-.', label="2 fix")
plt.semilogy(spectrums_fix_bit["lp_rff_budget_1024_nbit_1_seed_1"], '-.', label="1 fix")
plt.grid()
plt.legend()
plt.ylim([1e-5, 1e3])
plt.xlim([0, 2048])
plt.show()


# In[41]:


n_top_comp = 0


# In[42]:


rff_center = rff - np.mean(rff, axis=0)
main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
dy_range = np.max(residual, axis=0) - np.min(residual, axis=0)
plt.bar(np.arange(dy_range.size), dy_range)


# In[33]:


n_top_comp = 100


# In[38]:


rff_center = rff - np.mean(rff, axis=0)
main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
dy_range = np.max(residual, axis=0) - np.min(residual, axis=0)
plt.bar(np.arange(dy_range.size), dy_range)


# In[39]:


n_top_comp = 1000


# In[40]:


rff_center = rff - np.mean(rff, axis=0)
main_comp = np.dot(rff_center, U[:, 0:n_top_comp])
residual = rff_center - np.dot(main_comp, U[:, 0:n_top_comp].T)
dy_range = np.max(residual, axis=0) - np.min(residual, axis=0)
plt.bar(np.arange(dy_range.size), dy_range)

