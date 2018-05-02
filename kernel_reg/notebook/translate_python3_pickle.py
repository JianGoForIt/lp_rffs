import _pickle as cp
import os

general_folder = "/dfs/scratch0/zjian/lp_kernel/closeness/regression_real_setting/"
for folder in next(os.walk(general_folder))[1]:
    print(folder)
    if not os.path.isfile(general_folder + folder + "/metric_sample_train.txt"):
        continue	
    with open(general_folder + folder + "/metric_sample_train.txt", "rb") as f:
        w = cp.load(f)
    cp.dump(w, open(general_folder + folder + "/metric_sample_train_py2.txt","wb"), protocol=2)


