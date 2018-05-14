import os, sys
import math
from copy import deepcopy

# example
# full run lp rff lm halp: python launch_jobs_lp_rff_large_exp_lm_halp_mix_prec.py timit lm_halp/first_round_sweep starcluster without_metric cuda -1 dryrun early_stop &

#dataset = "census"
#exp_name = "nystrom_vs_rff"
#approx_type = "rff"
dataset = sys.argv[1]
exp_name = sys.argv[2]
cluster = sys.argv[3]  # starcluster / dawn
do_metric = sys.argv[4] # with_metric / without_metric
do_cuda = sys.argv[5]
n_subsample = sys.argv[6]
run_option = sys.argv[7]
early_stop = sys.argv[8]
model_nbit_list = [8]
feat_nbit_list = [4]
#approx_type = sys.argv[3]

# /dfs/scratch0/zjian/data/lp_kernel_data/census
if cluster == "starcluster":
    template = "python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py --model=unk --minibatch=250 --l2_reg=unk \
        --kernel_sigma=unk --n_fp_rff=unk --random_seed=unk --learning_rate=unk \
        --data_path=unk --opt=unk --epoch=unk --halp_mu=unk --halp_epoch_T=unk \
        --save_path=unk --approx_type=unk --n_bit_feat=unk --n_bit_model=unk"

else:
    template = "python /lfs/1/zjian/lp_kernel/lp_kernel/models/run_model.py --model=unk --minibatch=250 --l2_reg=unk \
        --kernel_sigma=unk --n_fp_rff=unk --random_seed=unk --learning_rate=unk \
        --data_path=unk --opt=unk --epoch=unk --halp_mu=unk --halp_epoch_T=unk \
        --save_path=unk --approx_type=unk --n_bit_feat=unk --n_bit_model=unk"


if dataset == "census":
    model = "ridge_regression"
    l2_reg_list = [0, ]
    kernel_sigma = math.sqrt(1.0/0.0006/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
    #n_fp_rff_list = [20000, ]
    seed_list = [1, 2, 3]
    lr_list = [0.5]
    #lr_list = [0.5, 0.1, 0.05, 0.01, 1.0]
    
elif dataset == "covtype":
    model = "logistic_regression"
    l2_reg_list = [0, ]
    kernel_sigma = math.sqrt(1.0/0.6/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
    #n_fp_rff_list = [1250, 10000, 100000]
    #n_fp_rff_list = [2500, 20000, 200000]
    #n_fp_rff_list = [5000, 50000, 400000]
    #n_fp_rff_list = [20000, ] # to simulate exact kernel approach
    seed_list = [1, 2, 3]
    #lr_list = [0.5, 1.0, 5.0, 10.0, 50.0]
    lr_list = [50.0]    

elif dataset == "yearpred":
    model = "ridge_regression"
    l2_reg_list = [0, ]
    kernel_sigma = math.sqrt(1.0/0.01/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
    #n_fp_rff_list = [1250, 10000, 100000]
    #n_fp_rff_list = [2500, 20000, 200000]
    #n_fp_rff_list = [5000, 50000, 400000]
    #n_fp_rff_list = [20000, ] # to simulate exact kernel approach
    seed_list = [1,2,3]
    #lr_list = [0.5, 1.0, 5.0, 10.0, 50.0]
    lr_list = [0.5]

elif dataset == "timit":
    model = "logistic_regression"
    l2_reg_list = [0, ]
    kernel_sigma = math.sqrt(1.0/0.0015/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
    #n_fp_rff_list = [1250, 10000, 100000]
    #n_fp_rff_list = [2500, 20000, 200000]
    #n_fp_rff_list = [5000, 50000, 400000]
    #n_fp_rff_list = [20000, ] # to simulate exact kernel approach
    halp_mu_list = [1e-3, 1e-2, 1e-1, 1e0]
    seed_list = [2, 3,]
    #lr_list = [1.0, 5.0, 10.0, 50.0, 100.0]
    lr_list = [100.0]

else:
    raise Exception("dataset not supported")

data_path = "/dfs/scratch0/zjian/data/lp_kernel_data/" + dataset
opt = "lm_halp"
epoch = 100 
halp_T = 1.0
save_path = "/dfs/scratch0/zjian/lp_kernel/" + exp_name + "/" + dataset

cnt = 0
for seed in seed_list:
	for l2_reg in l2_reg_list:
		for n_fp_rff in n_fp_rff_list:
			for lr in lr_list:
				for nbit_model, nbit_feat in zip(model_nbit_list, feat_nbit_list):
					for halp_mu in halp_mu_list:
						for approx_type in ["cir_rff"]:
							# if approx_type == "nystrom" and n_fp_rff > 20000:
							# 	continue
							if approx_type == "cir_rff" and n_fp_rff > 100000:
								continue
							save_suffix = "_type_" + approx_type + "_l2_reg_" + str(l2_reg) + "_n_fp_feat_" + str(n_fp_rff) \
							 	+ "_opt_" + opt + "_lr_" + str(lr) + "_model_nbit_" + str(nbit_model) + "_feat_n_bit_" + str(nbit_feat)  + "_halp_mu_" + str(halp_mu) + "_halp_T_" + str(halp_T) + "_seed_" + str(seed)
							command = deepcopy(template)
							command = command.replace("--model=unk", "--model="+model)
							command = command.replace("--l2_reg=unk", "--l2_reg="+str(l2_reg) )
							command = command.replace("--kernel_sigma=unk", "--kernel_sigma="+str(kernel_sigma) )
							command = command.replace("--n_fp_rff=unk", "--n_fp_rff="+str(n_fp_rff) )
							command = command.replace("--random_seed=unk", "--random_seed="+str(seed) )
							command = command.replace("--learning_rate=unk", "--learning_rate="+str(lr) )
							command = command.replace("--data_path=unk", "--data_path="+str(data_path) )
							command = command.replace("--opt=unk", "--opt="+opt)
							command = command.replace("--epoch=unk", "--epoch="+str(epoch) )
							command = command.replace("--save_path=unk", "--save_path="+save_path + save_suffix)
							command = command.replace("--approx_type=unk", "--approx_type="+approx_type)
							command = command.replace("--n_bit_feat=unk", "--n_bit_feat=" + str(nbit_feat) )
							command = command.replace("--n_bit_model=unk", "--n_bit_model=" + str(nbit_model) )
							command = command.replace("--halp_mu=unk", "--halp_mu=" + str(halp_mu) )
							command = command.replace("--halp_epoch_T=unk", "--halp_epoch_T="+str(halp_T))
							if do_metric == "with_metric":
								command += " --collect_sample_metrics"
							if do_cuda == "cuda":
								command += " --cuda"
							if int(n_subsample) > 0:
								command += " --n_sample=" + str(n_subsample)
							if early_stop == "no_early_stop":
								command += " --fixed_epoch_number"
							os.system("mkdir -p " + save_path + save_suffix)
							if cluster == "starcluster":
								command = "cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && " + command
							else:
								command = "cd /lfs/1/zjian/lp_kernel/lp_kernel/models && " + command
							f = open(save_path + save_suffix + "/job.sh", "w")
							f.write(command)
							f.close()
							if cluster == "starcluster":
								launch_command = "qsub -V " \
								+ " -o " + save_path + save_suffix + "/run.log " \
		                                	        + " -e " + save_path + save_suffix + "/run.err " + save_path + save_suffix + "/job.sh"
							else:
								launch_command = "bash " + save_path + save_suffix + "/job.sh"
							if run_option == "dryrun":
								print(launch_command)
							else:
								print(launch_command)	
								os.system(launch_command)

							cnt += 1
				#exit(0)
print(cnt, "jobs submitted!")

