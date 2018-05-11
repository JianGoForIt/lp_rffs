import numpy as np
import os, sys
import cPickle as cp
from copy import deepcopy
import matplotlib.pyplot as plt

#def get_results_for_one_precision(n_rff_feat, general_folder_measurement, 
#                                  general_folder_performance, general_folder_delta,
#                                  folder_pattern, seeds=[1,], min_best=True):
#    f_norm_list_rff = []
#    l2_loss_list_rff = []
#    delta_list_rff = []
#    
#    for seed in seeds:
#        f_norm_list = []
#        l2_loss_list = []
#        delta_list = []
#        for n_feat in n_rff_feat:
#            subfolder_name = deepcopy(folder_pattern)
#            if "n_fp_feat_unk" in subfolder_name:
#                subfolder_name = subfolder_name.replace("n_fp_feat_unk", "n_fp_feat_" + str(n_feat) )
#            else:
#                subfolder_name = subfolder_name.replace("n_feat_unk", "n_feat_" + str(n_feat) )
#            subfolder_name = subfolder_name.replace("seed_unk", "seed_" + str(seed) )
#            folder_name = general_folder_performance + "/" + subfolder_name
#            file_name = "eval_metric.txt"
#            l2_loss = get_performance_metric(folder_name, file_name, min_best=min_best)
#            l2_loss_list.append(l2_loss)
#            
#            folder_name = general_folder_delta + "/" + subfolder_name
#            file_name = "metric_sample_eval_py2.txt"
#            if not os.path.isfile(folder_name + "/" + file_name):
#                file_name = "metric_sample_eval.txt"
#            metric_name = "Delta"
#            delta = get_measurement_metric(folder_name, file_name, metric_name)
#            delta_list.append(delta)
#            
#            file_name = "metric_sample_eval_py2.txt"
#            if not os.path.isfile(folder_name + "/" + file_name):
#                file_name = "metric_sample_eval.txt"
#            metric_name = "F_norm_error"
#            f_norm_error = get_measurement_metric(folder_name, file_name, metric_name)
#            f_norm_list.append(f_norm_error)
#            
#        
##         print(l2_loss_list)
#        
#        f_norm_list_rff.append(np.array(deepcopy(f_norm_list) ) )
#        l2_loss_list_rff.append(np.array(deepcopy(l2_loss_list) ) )
#        delta_list_rff.append(np.array(deepcopy(delta_list) ) )
#        
#    f_norm_list_rff = average_results_array(f_norm_list_rff)
#    l2_loss_list_rff = average_results_array(l2_loss_list_rff)
#    delta_list_rff = average_results_array(delta_list_rff)
##     memory_list_rff = np.array( [rff_mem_func(n_feat) for n_feat in n_rff_feat] )
#    return f_norm_list_rff, l2_loss_list_rff, delta_list_rff#, memory_list_rff

def plot_figure(data_list, color_dict, ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
        average_y = average_results_array(y)
        std_y = std_results_array(y)
        if ave_x:
            x = average_results_array(x)
        plt.errorbar(x, average_y, yerr=std_y, label=label, markerfacecolor='none', fmt="-o", capsize=3, capthick=2, color=color_dict[label] )
    plt.grid()

def get_results_for_one_precision(n_rff_feat, general_folder_measurement, 
                                  general_folder_performance, general_folder_delta,
                                  folder_pattern, seeds=[1,], min_best=True, y_reverse=False):
    f_norm_list_rff = []
    l2_loss_list_rff = []
    delta_list_rff = []
    
    for seed in seeds:
        f_norm_list = []
        l2_loss_list = []
        delta_list = []
        for n_feat in n_rff_feat:
            subfolder_name = deepcopy(folder_pattern)
            if "n_fp_feat_unk" in subfolder_name:
                subfolder_name = subfolder_name.replace("n_fp_feat_unk", "n_fp_feat_" + str(n_feat) )
            else:
                subfolder_name = subfolder_name.replace("n_feat_unk", "n_feat_" + str(n_feat) )
            subfolder_name = subfolder_name.replace("seed_unk", "seed_" + str(seed) )
            folder_name = general_folder_performance + "/" + subfolder_name
            file_name = "eval_metric.txt"
            l2_loss = get_performance_metric(folder_name, file_name, min_best=min_best)
            l2_loss_list.append(l2_loss)
                        
            folder_name = general_folder_delta + "/" + subfolder_name
            if not os.path.exists(folder_name):
                folder_name = folder_name.replace("_opt_sgd_lr_10_", "_")
            file_name = "metric_sample_eval_py2.txt"
            if not os.path.isfile(folder_name + "/" + file_name):
                file_name = "metric_sample_eval.txt"
            metric_name = "Delta"
            delta = get_measurement_metric(folder_name, file_name, metric_name)
            delta_list.append(delta)
            
            file_name = "metric_sample_eval_py2.txt"
            if not os.path.isfile(folder_name + "/" + file_name):
                file_name = "metric_sample_eval.txt"
            metric_name = "F_norm_error"
            f_norm_error = get_measurement_metric(folder_name, file_name, metric_name)
            f_norm_list.append(f_norm_error)
        
        f_norm_list_rff.append(np.array(deepcopy(f_norm_list) ) )
        l2_loss_list_rff.append(1.0 - np.array(deepcopy(l2_loss_list) ) )
        delta_list_rff.append(np.array(deepcopy(delta_list) ) )
        
#     f_norm_list_rff = average_results_array(f_norm_list_rff)
#     l2_loss_list_rff = average_results_array(l2_loss_list_rff)
#     delta_list_rff = average_results_array(delta_list_rff)
#     memory_list_rff = np.array( [rff_mem_func(n_feat) for n_feat in n_rff_feat] )
    return f_norm_list_rff, l2_loss_list_rff, delta_list_rff#, memory_list_rff

def get_closeness(spectrum, spectrum_baseline, lamb):
    return np.linalg.norm(spectrum / (spectrum + lamb) - spectrum_baseline / (spectrum_baseline + lamb) )**2 

#def get_log_closeness(spectrum, spectrum_baseline, lamb):
#    return np.linalg.norm(np.log(spectrum + lamb) - np.log(spectrum_baseline + lamb) )**2

def get_log_closeness(spectrum, spectrum_baseline, lamb):
#     print "right version"
    assert np.all(spectrum >= 0.0)
    assert np.all(spectrum_baseline >= 0.0)
    val1 = (spectrum + lamb) / (spectrum_baseline + lamb)
    val2 = (spectrum_baseline + lamb) / (spectrum + lamb)
    Delta = np.max( (np.max(val1), np.max(val2) ) ) - 1.0
    log_closeness = np.linalg.norm(np.log(spectrum + lamb) - np.log(spectrum_baseline + lamb) )**2
#     print Delta, log_closeness
    return Delta

def get_spectrum(folder_name, file_name):
    if os.path.isfile(folder_name + "/" + file_name):
        with open(folder_name + "/" + file_name, "r") as f:
            spectrum = np.load(f)
        if spectrum[0] < spectrum[-1]:
            print spectrum[0], spectrum[-1]
            spectrum = spectrum[::-1]
    else:
        print folder_name, file_name, " is not found."
        spectrum = None 
    return spectrum

def get_performance_metric(folder_name, file_name, min_best=True):
#     print folder_name
    if os.path.isfile(folder_name + "/" + file_name):
        with open(folder_name + "/" + file_name, "r") as f:
            metrics = np.loadtxt(f)
        if min_best:
            metric = np.nanmin(metrics)
        else:
            metric = np.nanmax(metrics)
    else:
        print folder_name, file_name, " is not found."
        metric = None
    return metric

def get_measurement_metric(folder_name, file_name, metric_name):
#     print folder_name
    if os.path.isfile(folder_name + "/" + file_name):
	try:
            with open(folder_name + "/" + file_name, "r") as f:
                metrics = cp.load(f)
        except:
            with open(folder_name + "/" + file_name, "rb") as f:
                metrics = cp.load(f)
        if metric_name is not None:
            metric = metrics[metric_name]
        else:
            metric = metrics
    else:
        print folder_name, file_name, " is not found."
        metric = None
    return metric

def std_results_array(results_array):
    # average list of 1d np array results
#     print results_array
    results_array = [np.reshape(x, x.size) for x in results_array]
    results = np.vstack(results_array)
#     print results, results.shape
    return np.std(results, axis=0)

def average_results_array(results_array):
    # average list of 1d np array results
#     print results_array
    results_array = [np.reshape(x, x.size) for x in results_array]
    results = np.vstack(results_array)
#     print results, results.shape
    return np.mean(results, axis=0)

def get_nystrom_memory(n, m, r):
    '''
    let n be the number of kernel approximation features, m be the minibatch size.
    r is the original number of raw features. Assume for nystrom n landmark equals n kernel
    approximation features. 
    memory consumption for Nystrom
    n * n + n * r + m * n
    memory consumption for RFF
    n * r + m * n
    '''
    return n * r + n * n + m * n

def get_rff_memory(n, m, r):
    '''
    let n be the number of kernel approximation features, m be the minibatch size.
    r is the original number of raw features. Assume for nystrom n landmark equals n kernel
    approximation features. 
    memory consumption for Nystrom
    n * n + n * r + m * n
    memory consumption for RFF
    n * r + m * n + n
    '''
    return n * r + m * n + n

def get_cir_rff_memory(n, m, r, nbit):
    '''
    let n be the number of kernel approximation features, m be the minibatch size.
    r is the original number of raw features. 
    memory consumption for RFF:
    n (circulant projection size) + m * n * nbit / 64 + n
    '''
    return n + m * n * nbit / 32.0 + n

def get_nystrom_memory_with_model(n, m, r, c):
    '''
    let n be the number of kernel approximation features, m be the minibatch size.
    r is the original number of raw features. Assume for nystrom n landmark equals n kernel
    approximation features. c is the number of label classes. For regression c = 1
    '''
    return n * r + n * n + m * n + n * c

def get_rff_memory_with_model(n, m, r, c):
    '''
    let n be the number of kernel approximation features, m be the minibatch size.
    r is the original number of raw features. Assume for nystrom n landmark equals n kernel
    approximation features. c is the number of label classes. For regression c = 1
    '''
    return n * r + m * n + n + n * c

def get_cir_rff_memory_with_model(n, m, r, nbit, c):
    '''
    let n be the number of kernel approximation features, m be the minibatch size.
    r is the original number of raw features. c is the number of label classes. For regression c = 1
    memory consumption for RFF:
    n (circulant projection size) + m * n * nbit / 64 + n
    '''
    return n + m * n * nbit / 32.0 + n + n * c

def median_results_array(results_array):
    # average list of 1d np array results
#     print results_array
    results_array = [np.reshape(x, x.size) for x in results_array]
    results = np.vstack(results_array)
#     print results, results.shape
    return np.median(results, axis=0)
