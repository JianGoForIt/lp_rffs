import os, sys
import math
from copy import deepcopy

# example
#python launch_jobs_nystrom_vs_rff.py census nystrom_vs_rff dawn with_metric cuda -1 run &
#python launch_jobs_nystrom_vs_rff.py covtype nystrom_vs_rff_covtype_no_metric dawn without_metric cuda -1 run &
#for search lamda star for covtype in closeness experiments: python launch_jobs_nystrom_vs_rff.py covtype closeness/classification_real_setting starcluster without_metric cuda 20000 dryrun no_early_stop &
# for sweeping n feat with lambda star for covtype in closeness experiments: python launch_jobs_nystrom_vs_rff_delta.py covtype closeness/classification_real_setting dawn with_metric cuda 20000 dryrun no_early_stop &

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
#approx_type = sys.argv[3]

# /dfs/scratch0/zjian/data/lp_kernel_data/census
if cluster == "starcluster":
    template = "python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py --model=unk --minibatch=250 --l2_reg=unk \
        --kernel_sigma=unk --n_fp_rff=unk --random_seed=unk --learning_rate=unk \
        --data_path=unk --opt=unk --epoch=unk \
        --save_path=unk --approx_type=unk --do_fp_feat --exit_after_collect_metric"

else:
    template = "python /lfs/1/zjian/lp_kernel/lp_kernel/models/run_model.py --model=unk --minibatch=250 --l2_reg=unk \
        --kernel_sigma=unk --n_fp_rff=unk --random_seed=unk --learning_rate=unk \
        --data_path=unk --opt=unk --epoch=unk \
        --save_path=unk --approx_type=unk --do_fp_feat --exit_after_collect_metric"


if dataset == "census":
    model = "ridge_regression"
    if do_metric == "with_metric":
        l2_reg_list = [0.0, ]
    else:
        l2_reg_list = [0, 1e-3, 1e-2]
    kernel_sigma = math.sqrt(1.0/0.0006/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000, 800000]
#    n_fp_rff_list = [1250, 5000, 20000, 100000, 400000, 1600000]
#    n_fp_rff_list = [2500, 10000, 50000, 200000, 800000]    
    seed_list = [1, 2, 3]
    if do_metric == "with_metric":
        lr_list = [0.5]
    else:
        lr_list = [0.5, 0.1, 1.0]

elif dataset == "covtype":
    model = "logistic_regression"
    if do_metric == "with_metric":
        l2_reg_list = [1e-5, ]
    else:
        l2_reg_list = [5e-7, 1e-6, ] #[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    kernel_sigma = math.sqrt(1.0/0.6/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
    #n_fp_rff_list = [1250, 10000, 100000]
    #n_fp_rff_list = [2500, 20000, 200000]
    #n_fp_rff_list = [5000, 50000, 400000]
    #n_fp_rff_list = [20000, ] # to simulate exact kernel approach
    seed_list = [2,]
    if do_metric == "with_metric":
        lr_list = [10]
    else:
        lr_list = [10, 5, 50, 100]

elif dataset == "adult":
    model = "logistic_regression"
    if do_metric == "with_metric":
        l2_reg_list = [0.0, ]
    else:
        l2_reg_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    kernel_sigma = math.sqrt(1.0/0.1/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    #n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000]
    #n_fp_rff_list = [1250, 10000, 100000]
    #n_fp_rff_list = [2500, 20000, 200000]
    #n_fp_rff_list = [5000, 50000, 400000]
    n_fp_rff_list = [29304, ] # to simulate exact kernel approach
    seed_list = [1,]
    if do_metric == "with_metric":
        lr_list = [5]
    else:
        lr_list = [10, 5, 50, 1]
    print("right branch")

else:
    raise Exception("dataset not supported")

data_path = "/dfs/scratch0/zjian/data/lp_kernel_data/" + dataset
opt = "sgd"
epoch = 150
save_path = "/dfs/scratch0/zjian/lp_kernel/" + exp_name + "/" + dataset
if do_metric == "with_metric":
	save_path = save_path + "_with_metric"

cnt = 0
for seed in seed_list:
	for l2_reg in l2_reg_list:
		for n_fp_rff in n_fp_rff_list:
			for lr in lr_list:
				for approx_type in ["rff", "nystrom"]:
					if approx_type == "nystrom" and n_fp_rff > 20000:
						continue
					#if approx_type == "rff" and n_fp_rff < 200000:
					#	continue
					save_suffix = "_type_" + approx_type + "_l2_reg_" + str(l2_reg) + "_n_fp_feat_" + str(n_fp_rff) \
					 	+ "_opt_" + opt + "_lr_" + str(lr) + "_seed_" + str(seed)
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

