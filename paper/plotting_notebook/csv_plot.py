from matplotlib import pyplot as plt
import numpy as np
import csv
import os, sys

def plot_figure_with_error_bar(names, data, color_list):
    '''
    each column of data is a line
    the name follows the pattern like ['fp Nystrom-x', 'fp Nystrom-y', 'fp Nystrom-y_std', 'fp RFF-x', 'fp RFF-y', 'fp RFF-y_std']
    '''
    marker_list = ['o', 'v', '^', 's', 'h']
    for i in range(data.shape[1] // 3):
        idx = i * 3
        label = names[idx].split("-")[0]
        print "label ", names[idx], label
        x = data[:, idx]
        average_y = data[:, idx + 1]
        std_y = data[:, idx + 2]
#         print x, average_y, std_y   
        plt.errorbar(x, average_y, yerr=std_y, label=label, marker=marker_list[i], markeredgecolor=color_list[i % len(color_list)], fmt="-o", linewidth=1, capsize=3.5, capthick=1, color=colors[i % len(color_list)])

def set_fig_xtick(values, labels, fontsize):
    ax = plt.gca()
    ax.set_xticks(values)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    
def set_fig_ytick(values, labels, fontsize):
    ax = plt.gca()
    ax.set_yticks(values)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
