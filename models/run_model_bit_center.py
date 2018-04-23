import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from logistic_regression import LogisticRegression
from ridge_regression import RidgeRegression
import argparse
import sys, os
from copy import deepcopy
sys.path.append("../kernel_reg")
sys.path.append("../utils")
sys.path.append("../..")
from rff import RFF, GaussianKernel
from kernel_regressor import Quantizer
from data_loader import load_data
import halp
import halp.optim
import halp.quantize
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
args = parser.parse_args()


def train(args, model, epoch, train_loader, val_loader, optimizer, quantizer):
    for i, minibatch in enumerate(train_loader):
        X, Y = minibatch
        optimizer.zero_grad()
        if args.opt == "halp":
            # We need to add this function to models when we want to use SVRG
            def closure(data=X, target=Y):
                # print "test 123", type(data), type(target)
                data = kernel.get_cos_feat(data, dtype="float")
                if quantizer != None:
                    # print("halp use quantizer")
                    data = quantizer.quantize(data)
                if data.size(0) != target.size(0):
                    raise Exception("minibatch on data and target does not agree in closure")
                if not isinstance(data, torch.autograd.variable.Variable):
                    data = Variable(data, requires_grad=False)
                if not isinstance(target, torch.autograd.variable.Variable):
                    target = Variable(target, requires_grad=False)

                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                cost = model.forward(data, target)
                cost.backward()
                return cost
            loss = optimizer.step(closure)
            train_loss.append(loss[0].data.cpu().numpy() )
        else:
            X = kernel.get_cos_feat(X, dtype="float")
            if quantizer != None:
                # print("train use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            loss = model.forward(X, Y)
            train_loss.append(loss[0].data.cpu().numpy() )
            loss.backward()
            optimizer.step()
        # print("epoch ", epoch, "step", i, "loss", loss[0] )

    # perform evaluation
    sample_cnt = 0
    if args.model == "logistic_regression":
        correct_cnt = 0
        for i, minibatch in enumerate(val_loader):
            X, Y = minibatch
            X = kernel.get_cos_feat(X, dtype="float")
            if quantizer != None:
                # print("test use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            pred = model.predict(X)
            correct_cnt += np.sum(pred.reshape(pred.size, 1) == Y.data.cpu().numpy() )
            sample_cnt += pred.size
            # print correct_cnt, sample_cnt
        # print eval_acc
        eval_acc.append(correct_cnt / float(sample_cnt) )
        print("eval_acc at epoch ", epoch, "step", i, " iterations ", " acc ", eval_acc[-1] )
        return correct_cnt / float(sample_cnt)
    else:
        l2_accum = 0.0
        for i, minibatch in enumerate(val_loader):
            X, Y = minibatch
            X = kernel.get_cos_feat(X, dtype="float")
            if quantizer != None:
                # print("test use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            pred = model.predict(X)
            l2_accum += np.sum( (pred.reshape(pred.size, 1) \
                - Y.data.cpu().numpy().reshape(pred.size, 1) )**2)
            sample_cnt += pred.size
        eval_l2.append(l2_accum / float(sample_cnt) )
        print("eval_l2 at epoch ", epoch, "step", i, i, " iterations ", " loss ", np.sqrt(eval_l2[-1] ) )
        # return np.sqrt(l2_accum / float(sample_cnt) )
        return l2_accum / float(sample_cnt)



class ProgressMonitor(object):
    def __init__(self, init_lr=1.0, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=False, decay_thresh=0.99):
        self.lr = init_lr
        self.lr_decay_fac = lr_decay_fac
        self.min_lr = min_lr
        self.metric_history = []
        self.min_metric_better = min_metric_better
        self.best_model = None
        self.decay_thresh = decay_thresh
        self.prev_best = None
        self.drop_cnt = 0

    def end_of_epoch(self, metric, model, optimizer, epoch):
        if self.min_metric_better:
            model_is_better = (self.prev_best == None) or (metric <= self.prev_best)
        else:
            model_is_better = (self.prev_best == None) or (metric >= self.prev_best)

        if model_is_better:
            # save the best model
            self.best_model = deepcopy(model.state_dict() )
            print("saving best model with metric ", metric)
        else:
            # reverse to best model
            model.load_state_dict(deepcopy(self.best_model) )
            print("loading previous best model with metric ", self.prev_best)
        if (self.prev_best is not None) \
            and ( (self.min_metric_better and (metric > self.decay_thresh * self.prev_best) ) \
            or ( (not self.min_metric_better) and (self.prev_best > self.decay_thresh * metric) ) ):
            self.lr /= 2.0
            for g in optimizer.param_groups:
                g['lr'] = self.lr
            print("lr drop to ", self.lr)
            self.drop_cnt += 1

        if model_is_better:
            self.prev_best = metric

        self.metric_history.append(metric)
        if self.drop_cnt == 10:
            return True
        else:
            return False


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
    if use_cuda:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_val = X_val.cuda()
        Y_val = Y_val.cuda()

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
    if args.do_fp_feat == False:
        print("lp feature mode")
        assert args.n_bit_feat >= 1
        n_quantized_rff = int(np.floor(args.n_fp_rff / float(args.n_bit_feat) * 32.0) )
        kernel = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
        min_val = -np.sqrt(2.0/float(n_quantized_rff) )
        max_val = np.sqrt(2.0/float(n_quantized_rff) )
        quantizer = Quantizer(args.n_bit_feat, min_val, max_val, 
            rand_seed=args.random_seed, use_cuda=use_cuda)
        print("feature quantization scale, bit ", quantizer.scale, quantizer.nbit)
    else:
        print("fp feature mode")
        kernel = RFF(args.n_fp_rff, n_input_feat, kernel, rand_seed=args.random_seed)
        quantizer = None
    kernel.torch(cuda=use_cuda)

    if args.model == "logistic_regression":
        model = LogisticRegression(input_dim=kernel.n_feat, 
            n_class=n_class, reg_lambda=args.l2_reg)
        # model = LogisticRegression(input_dim=123, 
        #     n_class=n_class, reg_lambda=args.l2_reg)
    elif args.model == "ridge_regression":
        model = RidgeRegression(input_dim=kernel.n_feat, reg_lambda=args.l2_reg)
    if use_cuda:
        model.cuda()    

    # set up optimizer
    if args.opt == "sgd":
        print("using sgd optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.opt == "lpsgd":
        print("using lp sgd optimizer")
        optimizer = halp.optim.LPSGD(model.parameters(), lr=args.learning_rate, 
            scale_factor=args.scale_model, bits=args.n_bit_model)
        print("model quantization scale and bit ", optimizer._scale_factor, optimizer._bits)
    elif args.opt == "halp":
        print("using bit_center optimizer")
        optimizer = halp.optim.BitCenterLPSGD(model.parameters(), lr=args.learning_rate, 
            T=int(args.halp_epoch_T * X_train.size(0) / float(args.minibatch) ), 
            data_loader=train_loader, mu=args.halp_mu, bits=args.n_bit_model)
        print("model quantization, interval, mu, bit", optimizer.T, optimizer._mu, 
            optimizer._bits, optimizer._biased)
    else:
        raise Exception("optimizer not supported")
    # setup sgd training process
    train_loss = []
    if args.model == "logistic_regression":
        eval_acc = []
    else: 
        eval_l2 = []
    if args.model == "logistic_regression":
        monitor = ProgressMonitor(init_lr=args.learning_rate, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=False, decay_thresh=0.99)
    elif args.model == "ridge_regression":
        monitor = ProgressMonitor(init_lr=args.learning_rate, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=True, decay_thresh=0.99)
    else:
        raise Exception("model not supported!")
    for epoch in range(args.epoch):  
        # change learning rate
        metric = train(args, model, epoch, train_loader, val_loader, optimizer, quantizer)

        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        np.savetxt(args.save_path + "/train_loss.txt", train_loss)
        if args.model == "logistic_regression":
            np.savetxt(args.save_path + "/eval_metric.txt", eval_acc)
        elif args.model == "ridge_regression":
            np.savetxt(args.save_path + "/eval_metric.txt", eval_l2)
        else:
            raise Exception("model not supported!")

        # for param in optimizer._z:
        #     print param
        # early_stop = monitor.end_of_epoch(metric, model, optimizer, epoch)
        # if early_stop:
        #     break
