import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as cp
import sys
sys.path.append("../../utils/")
from plot_utils import get_colors

def get_value_from_folder(folder_name, file_name):
    with open(folder_name + "/" + file_name) as f:
        res = np.loadtxt(f)
    return res

def get_ave_value_from_multiple_seed(folder_name_template, file_name, general_folder, preprocessor):
    subdirs = [x[0] for x in os.walk(general_folder)] 
    results = []
    for subdir in subdirs:
        if folder_name_template in subdir:
#             print subdir
            results.append(get_value_from_folder(subdir, file_name) )
            if preprocessor is not None:
                results[-1] = preprocessor(results[-1] )
    ave = results[0]
    cnt = 1
    for res in results[1:]:
        try:
            ave += res
            cnt += 1
        except:
            pass
#         print res
    ave /= float(cnt)
#     print len(results)
    return ave

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def get_curve_best(curve_list, min_better=True, preprocessing=None):
    # the curve_list takes in list of tuple (curve, path)
    if preprocessing is not None:
        for i in range(len(curve_list) ):
            curve_list[i][0] = preprocessing(curve_list[i][0] )
    best_item_id = 0
    for i in range(1, len(curve_list) ):
        if min_better:
            if np.min(curve_list[i] ) < np.min(curve_list[best_item_id] ):
                best_item_id = i
        else:
            if np.max(curve_list[i] ) > np.max(curve_list[best_item_id] ):
                best_item_id = i
    return curve_list[best_item_id]
    
