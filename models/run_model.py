import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from logistic_regression import LogisticRegression
from ridge_regression import RidgeRegression
import argparse
import sys
sys.path.append("../kernel_reg")
sys.path.append("../utils")
from rff import RFF, GaussianKernel
from kernel_regressor import Quantizer
from data_loader import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="logistic_regression")
parser.add_argument("--minibatch", type=float, default=64)
parser.add_argument("--dataset", type=str, default="census")
parser.add_argument("--l2_reg", type=float, default=0.0)
parser.add_argument("--kernel_sigma", type=float, default=30.0)
parser.add_argument("--n_fp_rff", type=int, default=32)
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--n_bit_feat", type=int, default=32)
parser.add_argument("--do_fp", action="store_true")
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--data_path", type=str, default="../data/census/")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available() and args.cuda
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)
        # torch.cuda.manual_seed_all(args.seed)

    # load dataset
    X_train, X_val, Y_train, Y_val = load_data(args.data_path)
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    if args.model == "ridge_regression":
        Y_train = torch.FloatTensor(Y_train)        
        Y_val = torch.FloatTensor(Y_val)
    elif args.model == "logistic_regression":
        Y_train = Y_train.reshape( (Y_train.size) )
        Y_val = Y_val.reshape( (Y_val.size) )
        n_class = np.unique(np.hstack( (Y_train, Y_val) ) ).size
        Y_train = torch.LongTensor(Y_train)
        Y_val = torch.LongTensor(Y_val)
    else:
        raise Exception("model not supported")

    print "before ", type(X_train), type(Y_train)

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
    if args.do_fp == False:
        print("lp feature mode")
        assert args.n_bit_feat >= 1
        n_quantized_rff = int(np.floor(args.n_fp_rff / float(args.n_bit_feat) * 32.0) )
        kernel = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
        min_val = -np.sqrt(2.0/float(n_quantized_rff) )
        max_val = np.sqrt(2.0/float(n_quantized_rff) )
        quantizer = Quantizer(args.n_bit_feat, min_val, max_val, rand_seed=args.random_seed)
    else:
        print("fp feature mode")
        kernel = RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed)
    kernel.torch(cuda=use_cuda)

    if args.model == "logistic_regression":
        model = LogisticRegression(input_dim=kernel.n_feat, 
            n_class=n_class, reg_lambda=args.l2_reg)
    elif args.model == "ridge_regression":
        model = RidgeRegression(input_dim=kernel.n_feat, reg_lambda=args.l2_reg)

    # set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # setup sgd training process
    train_loss = []
    if args.model == "logistic_regression":
        eval_acc = []
    else:
        eval_l2 = []
    for epoch in range(args.epoch):
        for i, data in enumerate(train_loader):
            X, Y = data
            X = kernel.get_cos_feat(X, dtype="float")
            X = Variable(X)
            Y = Variable(Y)
            loss = model.forward(X, Y)
            train_loss.append(loss[0] )
            loss.backward()
            optimizer.step()
            print("epoch ", epoch, "step", i, "loss", loss[0] )

        sample_cnt = 0
        if args.model == "logistic_regression":
            correct_cnt = 0
            for i, data in enumerate(val_loader):
                X, Y = data
                X = kernel.get_cos_feat(X, dtype="float")
                X = Variable(X)
                Y = Variable(Y)
                pred = model.predict(X)
                correct_cnt += np.sum(pred == Y.data.cpu().numpy() )
                sample_cnt += pred.size
            eval_acc.append(correct_cnt / float(sample_cnt) )
            print("eval_acc at epoch ", epoch, "step", i, " iterations ", " acc ", eval_acc[-1] )
        else:
            l2_accum = 0.0
            for i, data in enumerate(val_loader):
                X, Y = data
                X = kernel.get_cos_feat(X, dtype="float")
                X = Variable(X)
                Y = Variable(Y)
                pred = model.predict(X)
                l2_accum += np.sum( (pred - Y.data.cpu().numpy() )**2)
                sample_cnt += pred.size
            eval_l2.append(l2_accum / float(sample_cnt) )
            print("eval_l2 at epoch ", epoch, "step", i, i, " iterations ", " loss ", np.sqrt(eval_l2[-1] ) )


        raw_input("waiting for")
