import _pickle as cp
import os

general_folder = "/dfs/scratch0/zjian/lp_kernel/lp_ensemble_nystrom/regression_real_setting/"
#general_folder = "/dfs/scratch0/zjian/lp_kernel/lp_rff/real_classification_heldout_spectral_norm/"
#general_folder = "/dfs/scratch0/zjian/lp_kernel/lp_rff/regression_real_setting_heldout_spectral_norm/"
for folder in next(os.walk(general_folder))[1]:
    #print(folder)
    if not os.path.isfile(general_folder + folder + "/metric_sample_eval.txt"):
        print("jump over ", general_folder + folder + "/metric_sample_eval.txt")
        continue
    #if "nystrom" in folder and ( ("seed_1" in folder) or ("seed_2" in folder) or ("seed_3" in folder)):
    #    pass
        #with open(general_folder + folder + "/metric_sample_eval.txt", "r") as f:
        #    w = cp.load(f)
    #else:
    with open(general_folder + folder + "/metric_sample_eval.txt", "rb") as f:
        w = cp.load(f)
#    print(folder)
    cp.dump(w, open(general_folder + folder + "/metric_sample_eval_py2.txt","wb"), protocol=2)

#general_folder = "/dfs/scratch0/zjian/lp_kernel/lp_rff/real_classification_heldout_spectral_norm/"
#for folder in next(os.walk(general_folder))[1]:
#    #print(folder)
#    if not os.path.isfile(general_folder + folder + "/metric_sample_eval.txt"):
#        print("jump over ", general_folder + folder + "/metric_sample_eval.txt")
#        continue
#    if "nystrom" in folder and ( ("seed_1" in folder) or ("seed_2" in folder) or ("seed_3" in folder)):
#        pass
#        #with open(general_folder + folder + "/metric_sample_eval.txt", "r") as f:
#        #    w = cp.load(f)
#    else:
#        with open(general_folder + folder + "/metric_sample_eval.txt", "rb") as f:
#            w = cp.load(f)
##    print(folder)
#        cp.dump(w, open(general_folder + folder + "/metric_sample_eval_py2.txt","wb"), protocol=2)
#
#
