import argparse
import numpy as np
import torch
import cPickle as cp
import sys, os
sys.path.append("../utils")

from data_loader import load_census_data
from rff import GaussianKernel, RFF
from pca_rff import PCA_RFF
from kernel_regressor import Quantizer, QuantizerAutoScale, KernelRidgeRegression

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
parser.add_argument("--pca_rff", action="store_true")
parser.add_argument("--pca_rff_mu", type=float, default=10.0)
parser.add_argument("--pca_rff_n_base_fp_feat", type=int, default=8192, 
  help="the number of rff feature for PCA RFF to setup;"\
  " the real memory budget is --n_fp_rff, it is the input to PCA_RFF setup function")
parser.add_argument("--pca_rff_auto_scale", action="store_true", 
  help="using percentile to auto decide the quantization dynamic range")
parser.add_argument("--pca_rff_perc", type=float,
  help="percentile to cut outliers in deciding the quantization dynamic range, \
  the value is 0.1 if we cut at 10 and 90 percentile.", default=0)
parser.add_argument("--fixed_design", action="store_true", 
  help="indicate doing fixed design runs on training data")
parser.add_argument("--fixed_design_data_sample_int", type=int, default=1,
  help="indicate the interval to sample train data to use for the fixed design run")
parser.add_argument("--fixed_design_noise_level", type=float, default=0.0,
  help="noise sigma in fixed design experiments")
args = parser.parse_args()


if __name__=="__main__":
  X_train, X_test, Y_train, Y_test = load_census_data(args.data_path)
  if args.fixed_design:
    print("fixed design mode")
    X_train = X_train[::args.fixed_design_data_sample_int, :]
    Y_train = Y_train[::args.fixed_design_data_sample_int]
    X_test = X_train
    Y_test = Y_train.copy()
    if args.fixed_design_noise_level != 0.0:
      # fixed_design_noise_level = \
      #   np.linalg.norm(Y_train) / np.sqrt(float(Y_train.size) ) * args.fixed_design_rel_noise_level
      Y_train += np.random.normal(scale=args.fixed_design_noise_level, size=Y_train.shape)
      Y_test += np.random.normal(scale=args.fixed_design_noise_level, size=Y_train.shape)

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
    print("fp mode")
    if args.pca_rff:
      print("pca rff model")
      kernel = PCA_RFF(args.pca_rff_n_base_fp_feat, n_input_feat, kernel, 
        rand_seed=args.random_seed, mu=args.pca_rff_mu)
      # pca rff need to do additional setup 
      kernel.setup(X_train, args.n_fp_rff)
      quantizer_train = None
      quantizer_test = None
      config_name = "fp_pca_rff_lambda_" + str(args.reg_lambda) + "_sigma_" \
        + str(args.sigma) + "_n_fp_rff_" + str(args.n_fp_rff) + \
        "_n_base_feat_" + str(args.pca_rff_n_base_fp_feat) + "_mu_" + str(args.pca_rff_mu)
    else:
      kernel = RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed)
      quantizer_train = None
      quantizer_test = None
      config_name = "fp_rff_lambda_" + str(args.reg_lambda) + "_sigma_" \
        + str(args.sigma) + "_n_fp_rff_" + str(args.n_fp_rff)
  else:
    if args.pca_rff:
      print("pca rff model")
      if args.pca_rff_auto_scale:
        print("using auto scale quantizer")
        quantizer_train = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=args.random_seed, percentile=args.pca_rff_perc)
      else:
        quantizer_train = Quantizer
      if not args.test_var_reduce:
        quantizer_test = quantizer_train
      else:
        quantizer_test = None
      kernel = PCA_RFF(args.pca_rff_n_base_fp_feat, n_input_feat, kernel, 
        rand_seed=args.random_seed, mu=args.pca_rff_mu)
      # pca rff need to do additional setup 
      kernel.setup(X_train, args.n_fp_rff)
      config_name = "lp_pca_rff_lambda_" + str(args.reg_lambda) + "_sigma_" \
        + str(args.sigma) + "_n_fp_rff_" + str(args.n_fp_rff) + "_nbit_" + str(args.n_bit) \
        + "_n_base_feat_" + str(args.pca_rff_n_base_fp_feat) + "_mu_" + str(args.pca_rff_mu)
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
  if args.fixed_design:
    U, S, _ = np.linalg.svd(regressor.kernel.rff_x1.cpu().numpy(), full_matrices=False)
    assert U.shape[0] == X_train.shape[0]
    with open(args.output_folder + "/kernel_eigen_vector.npy", "w") as f:
      np.save(f, U)
    with open(args.output_folder + "/kernel_eigen_value.npy", "w") as f:
      np.save(f, S**2)
  train_error = regressor.get_train_error()
  if args.pca_rff:
    regressor.kernel.test_mode()
  pred = regressor.predict(X_test, quantizer_train=quantizer_train, quantizer_test=quantizer_test)
  test_error = regressor.get_test_error(Y_test)
  # print("check test error and train error ", test_error, train_error)

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
  
  print("performance metrics: train l2/test l2/train approx error/test approx error", \
    train_error, test_error, kernel_mat_approx_error_train, kernel_mat_approx_error_test)
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






