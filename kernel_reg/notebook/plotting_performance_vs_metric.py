import numpy as np
import os, sys
import cPickle as cp

def get_closeness(spectrum, spectrum_baseline, lamb):
    return np.linalg.norm(spectrum / (spectrum + lamb) - spectrum_baseline / (spectrum_baseline + lamb) )**2 

def get_log_closeness(spectrum, spectrum_baseline, lamb):
    return np.linalg.norm(np.log(spectrum + lamb) - np.log(spectrum_baseline + lamb) )**2

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
        with open(folder_name + "/" + file_name, "r") as f:
            metrics = cp.load(f)
        metric = metrics[metric_name]
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
