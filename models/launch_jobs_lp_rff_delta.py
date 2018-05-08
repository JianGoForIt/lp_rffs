import os, sys
import math
from copy import deepcopy

# example
# for fp runs: python launch_jobs_nystrom_vs_rff_closed_form_sol_delta.py census delta/regression_real_settting dawn with_metric cpu dryrun 64 closed_form &
# for fp runs: python launch_jobs_nystrom_vs_rff_closed_form_sol_delta.py census delta/regression_real_settting dawn with_metric cpu dryrun 8 closed_form &

#dataset = "census"
#exp_name = "nystrom_vs_rff"
#approx_type = "rff"
dataset = sys.argv[1]
exp_name = sys.argv[2]
cluster = sys.argv[3]  # starcluster / dawn
do_metric = sys.argv[4] # with_metric / without_metric
do_cuda = sys.argv[5]
run_option = sys.argv[6]
nbit = sys.argv[7]
fixed_design = sys.argv[8]
closed_form = sys.argv[9]

# /dfs/scratch0/zjian/data/lp_kernel_data/census
if cluster == "starcluster":
	template = "python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py --model=ridge_regression \
		--l2_reg=unk  --kernel_sigma=unk --random_seed=unk \
		--data_path=unk --save_path=unk --approx_type=unk --n_fp_rff=unk"
else:
	template = "python /lfs/1/zjian/lp_kernel/lp_kernel/models/run_model.py --model=ridge_regression \
  		--l2_reg=unk  --kernel_sigma=unk --random_seed=unk \
  		--data_path=unk --save_path=unk --approx_type=unk --n_fp_rff=unk"



if dataset == "census":
    model = "ridge_regression"
    l2_reg_list = [5e-4, ]
    kernel_sigma = math.sqrt(1.0/0.0006/2.0)
    #n_fp_nystrom_list= [1250, 2500, 5000, 10000, 20000]
    # Note the below is the number of 
    n_rff_feat_dict = {"64": [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
                   "16": [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
                   "8": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
                   "4": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000],
                   "2": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000],
                   "1": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000], }
    n_fp_rff_list = n_rff_feat_dict[nbit]
#    n_fp_rff_list = [1250, 5000, 20000, 100000, 400000, 1600000]
#    n_fp_rff_list = [2500, 10000, 50000, 200000, 800000]    
    seed_list = [1,2,3,4,5]

data_path = "/dfs/scratch0/zjian/data/lp_kernel_data/" + dataset
save_path = "/dfs/scratch0/zjian/lp_kernel/" + exp_name + "/" + dataset

cnt = 0
for seed in seed_list:
	for l2_reg in l2_reg_list:
		for n_fp_rff in n_fp_rff_list:
			for approx_type in ["cir_rff", "rff"]:
				if approx_type == "cir_rff" and n_fp_rff > 200000:
					continue
				if approx_type == "rff" and (nbit != "64" or n_fp_rff > 200000):
					continue
				save_suffix = "_type_" + approx_type + "_l2_reg_" + str(l2_reg) + "_n_fp_feat_" + str(n_fp_rff) + "_seed_" + str(seed)
				command = deepcopy(template)
				command = command.replace("--l2_reg=unk", "--l2_reg="+str(l2_reg) )
				command = command.replace("--kernel_sigma=unk", "--kernel_sigma="+str(kernel_sigma) )
				command = command.replace("--n_fp_rff=unk", "--n_fp_rff="+str(n_fp_rff) )
				command = command.replace("--random_seed=unk", "--random_seed="+str(seed) )
				command = command.replace("--data_path=unk", "--data_path="+str(data_path) )
				command = command.replace("--save_path=unk", "--save_path="+save_path + save_suffix)
				command = command.replace("--approx_type=unk", "--approx_type="+approx_type)
				if nbit == "64":
					# do full precision
					command = command + " --do_fp_feat "
				else:
					command = command + " --n_bit_feat=" + str(nbit)
				if closed_form == "closed_form_sol":
					command = command + " --closed_form_sol"
				else:
					command = command + " --opt=sgd"
				if do_metric == "with_metric":
					command += " --collect_sample_metrics"
				if do_cuda == "cuda":
					command += " --cuda"
				os.system("mkdir -p " + save_path + save_suffix)
				if cluster == "starcluster":
					command = "cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && " + command
				else:
					command = "cd /lfs/1/zjian/lp_kernel/lp_kernel/models && " + command
				f = open(save_path + save_suffix + "/job.sh", "w")
				f.write(command)
				f.close()
				if cluster == "starcluster":
					# distinguish the log from runs calculating closeness related metrics or not
					launch_command = "qsub -V " \
					+ " -o " + save_path + save_suffix + "/run.log" \
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

