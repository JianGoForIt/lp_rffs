import torch
import numpy as np
from logistic_regression import LogisticRegression
from ridge_regression import RidgeRegression
import sys
sys.path.append("../kernel_reg")
from kernel_regressor import Quantizer


parser = argparse.ArgumentParser()
parser.add_argument("--classifier", type=str, default="logistic_regression")
parser.add_argument("--minibatch", type=float, default=64)
parser.add_argument("--dataset", type=str, default="census")
parser.add_argument("--l2_reg", type=float, default=0.0)
parser.add_argument("--n_fp_rff", type=int, default=32)
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--n_bit_feat", type=int, default=32)
parser.add_argument("--do_fp", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
	# load dataset
	if args.dataset == "census":
		X_train, X_test, Y_train, Y_test = load_census_data(args.data_path)
		Y_train = torch.Tensor(Y_train)
		Y_test = torch.Tensor(Y_test)
	else:
		raise Exception("Dataset not supported")

	# setup gaussian kernel
	n_input_feat = X_train.shape[1]
	kernel = GaussianKernel(sigma=args.sigma)
	if args.do_fp == False:
		print("lp feature mode")
		assert args.n_bit_feat >= 1
		n_quantized_rff = int(np.floor(args.n_fp_rff / float(args.n_bit) * 32.0) )
		kernel = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
		X_train = kernel.get_cos_feat(X_train)
		X_test = kernel.get_cos_feat(X_test)
		min_val = -np.sqrt(2.0/float(n_quantized_rff) )
		max_val = np.sqrt(2.0/float(n_quantized_rff) )
		quantizer = Quantizer(args.n_bit, min_val, max_val, rand_seed=args.random_seed)
		X_train = quantizer.quantize(X_train)
		X_test = quantizer.quantize(X_test)
	else:
		print("fp feature mode")
		kernel = RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed)
		X_train = kernel.get_cos_feat(X_train)
		X_test = kernel.get_cos_feat(X_test)

	
	# setup dataloader  
	train_data = torch.utils.data.TensorDataset(X_train, Y_train)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.minibatch, shuffle=False)

	# setup sgd training process
	

