import argparse
import numpy as np
import torch
import cPickle as cp
import sys, os
sys.path.append("../utils")

from data_loader import load_census_data
from rff import GaussianKernel, RFF
from kernel_regressor import Quantizer, KernelRidgeRegression

parser = argparse.ArgumentParser()
parser.add_argument("--n_fp_rff", type=int, default=100)
parser.add_argument("--n_bit", type=int, default=32)
parser.add_argument("--do_fp", action="store_true")
parser.add_argument("--exact_kernel", action="store_true")
parser.add_argument("--reg_lambda", type=float, default=0.001)
parser.add_argument("--sigma", type=float, default=1.0, help="the kernel width")
parser.add_argument("--data_path", type=str, default="../../data/census/")
parser.add_argument("--output_folder", type=str, default="./output.pkl")
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--test_var_reduce", action="store_true")
args = parser.parse_args()


if __name__=="__main__":
  X_train, X_test, Y_train, Y_test = load_census_data(args.data_path)
  kernel = GaussianKernel(sigma=args.sigma)
  kernel_mat_exact_train = kernel.get_kernel_matrix(X_train, X_train)
  kernel_mat_exact_test = kernel.get_kernel_matrix(X_test, X_train)
  n_input_feat = X_train.shape[1]
  assert X_train.shape[1] == X_test.shape[1]
  if args.exact_kernel:
    kernel = kernel
    quantizer_train = None
    quantizer_test = None
    config_name = "exact_kernel_lambda_" + str(args.reg_lambda) + "_sigma_" + str(args.sigma)
  elif args.do_fp:
    kernel = RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed)
    quantizer_train = None
    quantizer_test = None
    config_name = "fp_rff_lambda_" + str(args.reg_lambda) + "_sigma_" \
      + str(args.sigma) + "_n_fp_rff_" + str(args.n_fp_rff)
  else:
    n_quantized_rff = int(np.floor(args.n_fp_rff / float(args.n_bit) * 32.0) )
    min_val = -np.sqrt(2.0/float(n_quantized_rff) )
    max_val = np.sqrt(2.0/float(n_quantized_rff) )
    quantizer_train = Quantizer(args.n_bit, min_val, max_val, rand_seed=args.random_seed)
    if not args.test_var_reduce:
      quantizer_test = quantizer_train
    else:
      quantizer_test = None
    kernel = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
    config_name = "lp_rff_lambda_" + str(args.reg_lambda) + "_sigma_" \
      + str(args.sigma) + "_n_fp_rff_" + str(args.n_fp_rff) + "_nbit_" + str(args.n_bit) 

  regressor = KernelRidgeRegression(kernel, reg_lambda=args.reg_lambda)
  print("start to do regression!")
  # print("test quantizer", quantizer)
  regressor.fit(X_train, Y_train, quantizer=quantizer_train)
  print("finish regression!")
  train_error = regressor.get_train_error()
  pred = regressor.predict(X_test, quantizer_train=quantizer_train, quantizer_test=quantizer_test)
  test_error = regressor.get_test_error(Y_test)
  print("check test error and train error ", test_error, train_error)

  # get kernel approximation error 
  kernel_mat_approx_error_train = torch.sum( (regressor.kernel_mat - kernel_mat_exact_train)**2)
  kernel_mat_approx_error_test = torch.sum( (regressor.kernel_mat_pred - kernel_mat_exact_test)**2)
  # print "F norm exact kernel ", torch.sum(kernel_mat_exact_train**2)

  # dict_res = {}
  # if os.path.isfile(args.output_folder):
  #   with open(args.output_folder, "r") as f:
  #     dict_res.update(cp.load(f) )
  # dict_res[config_name] = {"train_l2_error": train_error,
  #   "test_l2_error": test_error, "train_approx_error": kernel_mat_approx_error_train,
  #   "test_approx_error": kernel_mat_approx_error_test}
  # with open(args.output_folder, "w") as f:
  #   cp.dump(dict_res, f)
  
  if not os.path.isdir(args.output_folder):
    os.makedirs(args.output_folder)
  dict_res = {"train_l2_error": train_error,
    "test_l2_error": test_error, "train_approx_error": kernel_mat_approx_error_train,
    "test_approx_error": kernel_mat_approx_error_test}
  with open(args.output_folder + "/results.pkl", "w") as f:
    cp.dump(dict_res, f)


  # dict_res = {}
  # if os.path.isfile(args.output_folder):
  #   with open(args.output_folder, "r") as f:
  #     dict_res.update(cp.load(f) )
  # print(dict_res)
  
  # with open("./test/lambda_0.01_seed_2_fp_rff_nbit_16/results.pkl", "r") as f:
  #     dict_res = cp.load(f)
  # print(dict_res)






