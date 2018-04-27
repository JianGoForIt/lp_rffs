import os, sys
import math
from copy import deepcopy

# /dfs/scratch0/zjian/data/lp_kernel_data/census
template = "python run_model.py --model=unk --minibatch=250 --l2_reg=unk \
  --kernel_sigma=unk --n_fp_rff=unk --random_seed=unk --learning_rate=unk \
  --data_path=unk --opt=unk --epoch=unk \
  --save_path=unk --approx_type=unk --cuda"

dataset = "census"
model = "ridge_regression"
l2_reg_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1e0, 1e1]
kernel_sigma = math.sqrt(1.0/0.0006/2.0)
n_fp_rff_list = [1250, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 400000, 800000]
seed_list = [1, 2, 3]
lr_list = [1.0, 0.5, 0.1]
data_path = "/dfs/scratch0/zjian/data/lp_kernel_data/" + dataset
opt = "sgd"
epoch = 300
save_path = "/dfs/scratch0/zjian/lp_kernel_results/" + dataset
approx_type = "rff"

for seed in seed_list:
	for l2_reg in l2_reg_list:
		for n_fp_rff in n_fp_rff_list:
			for lr in lr_list:
				save_suffix = "_l2_reg_" + str(l2_reg) + "_n_fp_rff_" + str(n_fp_rff) \
				 	+ "_opt_" + opt + "_lr_" + str(lr) + "_seed_" + str(seed)
				command = deepcopy(template)
				command = command.replace("--model=unk", "--model="+model)
				command = command.replace("--l2_reg=unk", "--l2_reg="+str(lr) )
				command = command.replace("--kernel_sigma=unk", "--kernel_sigma="+str(kernel_sigma) )
				command = command.replace("--n_fp_rff=unk", "--n_fp_rff="+str(n_fp_rff) )
				command = command.replace("--random_seed=unk", "--random_seed="+str(seed) )
				command = command.replace("--learning_rate=unk", "--learning_rate="+str(lr) )
				command = command.replace("--data_path=unk", "--data_path="+str(data_path) )
				command = command.replace("--opt=unk", "--opt="+opt)
				command = command.replace("--epoch=unk", "--epoch="+str(epoch) )
				command = command.replace("--save_path=unk", "--save_path="+save_path + save_suffix)
				command = command.replace("--approx_type=unk", "--approx_type="+approx_type)

				if sys.argv[1] == "dryrun":
					print command
				else:
					os.system(command)



