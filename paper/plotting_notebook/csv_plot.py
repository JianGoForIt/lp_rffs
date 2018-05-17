from matplotlib import pyplot as plt
import numpy as np
import csv
import os, sys

def save_csv_with_error_bar(data_list, file_name="./test/test.csv", ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    df_list = []
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
        average_y = average_results_array(y)
        std_y = std_results_array(y)
        if ave_x:
            x = average_results_array(x)
        x = np.array(x)
        average_y = np.array(average_y)
        std_y = np.array(std_y)
        df_list.append(pd.DataFrame(np.reshape(x, [x.size, 1] ), columns = [label + "-x" ] ) )
        df_list.append(pd.DataFrame(np.reshape(average_y, [average_y.size, 1] ), columns = [label + "-y" ] ) )
        df_list.append(pd.DataFrame(np.reshape(std_y, [std_y.size, 1] ), columns = [label + "-y_std" ] ) )
    pd.concat(df_list, axis=1).to_csv(file_name)


def csv_to_table(file_name, delimiter=',', row_headers=True):
    with open(file_name, 'r') as csv_file:  
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        groups = next(csv_reader)[1:]
        names = []
        data = []
        for row in csv_reader:
            if row_headers:
                names.append(row[0])
                # print([type(x) for x in row[1:] ])
                #print(row)
                #print()
                row2 = []
                for r in row[1:]:
                    if r == '':
                        row2.append(float('nan'))
                    else:
                        row2.append(float(r))
                data.append(list(row2))
                #data.append(list(map(lambda x: float(x), row[1:])))
            else:
                data.append(list(map(lambda x: float(x), row)))
    return groups, names, np.array(data)

def plot_figure_with_error_bar(names, data, color_list):
    '''
    each column of data is a line
    the name follows the pattern like ['fp Nystrom-x', 'fp Nystrom-y', 'fp Nystrom-y_std', 'fp RFF-x', 'fp RFF-y', 'fp RFF-y_std']
    '''
    marker_list = ['o', 'v', '^', 's', 'h', 'd', '+']
    for i in range(data.shape[1] // 3):
        idx = i * 3
        label = names[idx].split("-")[0]
        print "label ", names[idx], label
        x = data[:, idx]
        average_y = data[:, idx + 1]
        std_y = data[:, idx + 2]
#         print x, average_y, std_y   
        plt.errorbar(x, average_y, yerr=std_y, label=label, marker=marker_list[i], markeredgecolor=color_list[i % len(color_list)], fmt="-o", linewidth=1, capsize=3.5, capthick=1, color=color_list[i % len(color_list)])

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
