import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from logistic_regression import LogisticRegression
from ridge_regression import RidgeRegression
import argparse
import sys, os
import cPickle as cp
from copy import deepcopy
sys.path.append("../kernel_reg")
sys.path.append("../utils")
sys.path.append("../..")
from rff import RFF, GaussianKernel
from nystrom import Nystrom
from kernel_regressor import Quantizer
from data_loader import load_data
import halp
import halp.optim
import halp.quantize
from train_utils import train, evaluate, ProgressMonitor, get_sample_kernel_metrics
# import halp.optim
# from halp import LPSGD
# from halp import HALP

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="logistic_regression")
parser.add_argument("--minibatch", type=float, default=64)
# parser.add_argument("--dataset", type=str, default="census")
parser.add_argument("--l2_reg", type=float, default=0.0)
parser.add_argument("--kernel_sigma", type=float, default=30.0)
parser.add_argument("--n_fp_rff", type=int, default=32)
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--n_bit_feat", type=int, default=32)
parser.add_argument("--n_bit_model", type=int, default=32)
parser.add_argument("--scale_model", type=float, default=0.00001)
parser.add_argument("--do_fp_model", action="store_true")
parser.add_argument("--do_fp_feat", action="store_true")
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--data_path", type=str, default="../data/census/")
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--halp_mu", type=float, default=10.0)
parser.add_argument("--halp_epoch_T", type=float, default=1.0, 
    help="The # of epochs as interval of full gradient calculation")
parser.add_argument("--save_path", type=str, default="./test")
parser.add_argument("--approx_type", type=str, default="rff", help="specify using exact, rff or nystrom")
parser.add_argument("--collect_sample_metrics", action="store_true", 
    help="True if we want to collect metrics from the subsampled kernel matrix")
parser.add_argument("--n_measure_sample", type=int, default=20000, 
    help="samples for metric measurements, including approximation error and etc.")
args = parser.parse_args()



if __name__ == "__main__":
    use_cuda = torch.cuda.is_available() and args.cuda
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)
        # torch.cuda.manual_seed_all(args.seed)
    # load dataset
    X_train, X_val, Y_train, Y_val = load_data(args.data_path)
    X_train = torch.FloatTensor(X_train.astype(np.float32) )
    X_val = torch.FloatTensor(X_val.astype(np.float32) )
    if args.model == "ridge_regression":
        Y_train = torch.FloatTensor(Y_train.astype(np.float32) )        
        Y_val = torch.FloatTensor(Y_val.astype(np.float32) )
    elif args.model == "logistic_regression":
        Y_train = Y_train.reshape( (Y_train.size) )
        Y_val = Y_val.reshape( (Y_val.size) )
        n_class = np.unique(np.hstack( (Y_train, Y_val) ) ).size
        Y_train = torch.LongTensor(np.array(Y_train.tolist() ).reshape(Y_train.size, 1) )
        Y_val = torch.LongTensor(np.array(Y_val.tolist() ).reshape(Y_val.size, 1) )
    else:
        raise Exception("model not supported")
    # if use_cuda:
    #     X_train = X_train.cuda()
    #     Y_train = Y_train.cuda()
    #     X_val = X_val.cuda()
    #     Y_val = Y_val.cuda()

    # setup dataloader 
    train_data = \
        torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.minibatch, shuffle=False)
    val_data = \
        torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.minibatch, shuffle=False)

    # setup gaussian kernel
    n_input_feat = X_train.shape[1]
    kernel = GaussianKernel(sigma=args.kernel_sigma)  
    if args.approx_type == "exact":
        print("exact kernel mode")
        # raise Exception("SGD based exact kernel is not implemented yet!")
        kernel_approx = kernel
        quantizer = None
    elif args.approx_type == "nystrom":
        print("fp nystrom mode")
        kernel_approx = Nystrom(args.n_fp_rff, kernel=kernel, rand_seed=args.random_seed) 
        kernel_approx.setup(X_train) 
        quantizer = None
    elif args.approx_type == "rff" and args.do_fp_feat == False:
        print("lp rff feature mode")
        assert args.n_bit_feat >= 1
        n_quantized_rff = int(np.floor(args.n_fp_rff / float(args.n_bit_feat) * 32.0) )
        kernel_approx = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
        min_val = -np.sqrt(2.0/float(n_quantized_rff) )
        max_val = np.sqrt(2.0/float(n_quantized_rff) )
        quantizer = Quantizer(args.n_bit_feat, min_val, max_val, 
            rand_seed=args.random_seed, use_cuda=use_cuda)
        print("feature quantization scale, bit ", quantizer.scale, quantizer.nbit)
    elif args.approx_type == "rff" and args.do_fp_feat == True:
        print("fp rff feature mode")
        kernel_approx = RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed)
        quantizer = None
    else:
        raise Exception("kernel approximation type not specified!")
    kernel.torch(cuda=use_cuda)
    kernel_approx.torch(cuda=use_cuda)


    # construct model
    if args.model == "logistic_regression":
        model = LogisticRegression(input_dim=kernel_approx.n_feat, 
            n_class=n_class, reg_lambda=args.l2_reg)
        # model = LogisticRegression(input_dim=123, 
        #     n_class=n_class, reg_lambda=args.l2_reg)
    elif args.model == "ridge_regression":
        model = RidgeRegression(input_dim=kernel_approx.n_feat, reg_lambda=args.l2_reg)
    if use_cuda:
        model.cuda()    

    # set up optimizer
    if args.opt == "sgd":
        print("using sgd optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
    elif args.opt == "lpsgd":
        print("using lp sgd optimizer")
        optimizer = halp.optim.LPSGD(model.parameters(), lr=args.learning_rate, 
            scale_factor=args.scale_model, bits=args.n_bit_model, weight_decay=args.l2_reg)
        print("model quantization scale and bit ", optimizer._scale_factor, optimizer._bits)
    elif args.opt == "halp":
        print("using halp optimizer")
        optimizer = halp.optim.HALP(model.parameters(), lr=args.learning_rate, 
            T=int(args.halp_epoch_T * X_train.size(0) / float(args.minibatch) ), 
            data_loader=train_loader, mu=args.halp_mu, bits=args.n_bit_model, weight_decay=args.l2_reg)
        print("model quantization, interval, mu, bit", optimizer.T, optimizer._mu, 
            optimizer._bits, optimizer._biased)
    else:
        raise Exception("optimizer not supported")
    

    # collect metrics
    if args.collect_sample_metrics:
        print("start doing sample metric collection with ", args.n_measure_sample, " samples")
        metric_dict_sample_train, spectrum_sample_train = \
            get_sample_kernel_metrics(X_train, kernel, kernel_approx, quantizer, args.n_measure_sample)
        metric_dict_sample_val, spectrum_sample_val = \
            get_sample_kernel_metrics(X_val, kernel, kernel_approx, quantizer, args.n_measure_sample)    
        with open(args.save_path + "/metric_sample_train.txt", "w") as f:
            cp.dump(metric_dict_sample_train, f)
        np.save(args.save_path + "/spectrum_train.npy", spectrum_sample_train)
        with open(args.save_path + "/metric_sample_eval.txt", "w") as f:
            cp.dump(metric_dict_sample_val, f)
        np.save(args.save_path + "/spectrum_eval.npy", spectrum_sample_val)
        # print metric_dict_sample_train, metric_dict_sample_val
        # print spectrum_sample_train, spectrum_sample_val
        print("Sample metric collection done!")


    # setup sgd training process
    train_loss = []
    eval_metric = []
    if args.model == "logistic_regression":
        monitor = ProgressMonitor(init_lr=args.learning_rate, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=True, decay_thresh=0.999)
    elif args.model == "ridge_regression":
        monitor = ProgressMonitor(init_lr=args.learning_rate, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=True, decay_thresh=0.999)
    else:
        raise Exception("model not supported!")
    for epoch in range(args.epoch):  
        # train for one epoch
        loss_per_step = train(args, model, epoch, train_loader, optimizer, quantizer, kernel_approx)
        train_loss += loss_per_step
        # evaluate and save evaluate metric
        metric, monitor_signal = evaluate(args, model, epoch, val_loader, quantizer, kernel_approx)
        eval_metric.append(metric)

        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        np.savetxt(args.save_path + "/train_loss.txt", train_loss)
        np.savetxt(args.save_path + "/eval_metric.txt", eval_metric)
        
        # for param in optimizer._z:
        #     print param
        early_stop = monitor.end_of_epoch(monitor_signal, model, optimizer, epoch)
        if early_stop:
            break
