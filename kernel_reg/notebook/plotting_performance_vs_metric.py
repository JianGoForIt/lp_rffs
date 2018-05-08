import numpy as np
import os, sys
import cPickle as cp

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

def median_results_array(results_array):
    # average list of 1d np array results
#     print results_array
    results_array = [np.reshape(x, x.size) for x in results_array]
    results = np.vstack(results_array)
#     print results, results.shape
    return np.median(results, axis=0)
