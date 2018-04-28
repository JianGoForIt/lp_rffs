import os, sys
import math
from copy import deepcopy

# /dfs/scratch0/zjian/data/lp_kernel_data/census
template = "python /dfs/scratch0/zjian/lp_kernel/lp_kernel/models/run_model.py --model=unk --minibatch=250 --l2_reg=unk \
  --kernel_sigma=unk --n_fp_rff=unk --random_seed=unk --learning_rate=unk \
  --data_path=unk --opt=unk --epoch=unk \
  --save_path=unk --approx_type=unk --cuda"

#dataset = "census"
#exp_name = "nystrom_vs_rff"
#approx_type = "rff"
dataset = sys.argv[1]
exp_name = sys.argv[2]
approx_type = sys.argv[3]

if dataset == "census":
    model = "ridge_regression"
    l2_reg_list = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    kernel_sigma = math.sqrt(1.0/0.0006/2.0)
    n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000, 800000, 1600000]
    seed_list = [1, 2, 3]
    lr_list = [0.5, 0.1, 1.0]
    data_path = "/dfs/scratch0/zjian/data/lp_kernel_data/" + dataset
    opt = "sgd"
    epoch = 300
    save_path = "/dfs/scratch0/zjian/lp_kernel_results/" + exp_name + "/" + dataset

cnt = 0
for seed in seed_list:
	for l2_reg in l2_reg_list:
		for n_fp_rff in n_fp_rff_list:
			for lr in lr_list:
				save_suffix = "_l2_reg_" + str(l2_reg) + "_n_fp_rff_" + str(n_fp_rff) \
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
				os.system("mkdir -p " + save_path + save_suffix)
				command = "cd /dfs/scratch0/zjian/lp_kernel/lp_kernel/models && " + command
				f = open(save_path + save_suffix + "/job.sh", "w")
				f.write(command)
				f.close()
				launch_command = "qsub -V " \
					+ " -o " + save_path + save_suffix + "/run.log " \
                                        + " -e " + save_path + save_suffix + "/run.err " + save_path + save_suffix + "/job.sh"
				if sys.argv[4] == "dryrun":
					print(launch_command)
				else:
					print(launch_command)	
					os.system(launch_command)

				cnt += 1
print(cnt, "jobs submitted!")

