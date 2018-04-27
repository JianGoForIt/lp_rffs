import torch
from torch.autograd import Variable
import numpy as np

def train(args, model, epoch, train_loader, optimizer, quantizer, kernel):
    train_loss = []
    for i, minibatch in enumerate(train_loader):
        X, Y = minibatch
        optimizer.zero_grad()
        if args.opt == "halp":
            # We need to add this function to models when we want to use SVRG
            def closure(data=X, target=Y):
                # print "test 123", type(data), type(target)
                if args.approx_type == "rff":
                    data = kernel.get_cos_feat(data, dtype="float")
                elif args.approx_type == "nystrom":
                    data = kernel.get_feat(data)
                else:
                    raise Exception("kernel approximation type not supported!")
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
            if args.approx_type == "rff":
                X = kernel.get_cos_feat(X, dtype="float")
            elif args.approx_type == "nystrom":
                X = kernel.get_feat(X)
            else:
                raise Exception("kernel approximation type not supported!")
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
    return train_loss

def evaluate(args, model, epoch, val_loader, quantizer, kernel):
    # perform evaluation
    sample_cnt = 0
    if args.model == "logistic_regression":
        correct_cnt = 0
        for i, minibatch in enumerate(val_loader):
            X, Y = minibatch
            if args.approx_type == "rff":
                X = kernel.get_cos_feat(X, dtype="float")
            elif args.approx_type == "nystrom":
                X = kernel.get_feat(X)
            else:
                raise Exception("kernel approximation type not supported!")
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
        # eval_acc.append(correct_cnt / float(sample_cnt) )
        print("eval_acc at epoch ", epoch, "step", i, " iterations ", " acc ", correct_cnt / float(sample_cnt) )
        return correct_cnt / float(sample_cnt)
    else:
        l2_accum = 0.0
        for i, minibatch in enumerate(val_loader):
            X, Y = minibatch
            if args.approx_type == "rff":
                X = kernel.get_cos_feat(X, dtype="float")
            elif args.approx_type == "nystrom":
                X = kernel.get_feat(X)
            else:
                raise Exception("kernel approximation type not supported!")
            if quantizer != None:
                # print("test use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            pred = model.predict(X)
            l2_accum += np.sum( (pred.reshape(pred.size, 1) \
                - Y.data.cpu().numpy().reshape(pred.size, 1) )**2)
            sample_cnt += pred.size
        # eval_l2.append(l2_accum / float(sample_cnt) )
        print("eval_l2 at epoch ", epoch, "step", i, " iterations ", " loss ", np.sqrt(l2_accum / float(sample_cnt) ) )
        # return np.sqrt(l2_accum / float(sample_cnt) )
        return l2_accum / float(sample_cnt)


def sample_data(X, n_sample):
    '''
    X is in the shape of [n_sample, n_feat]
    '''
    perm = np.random.permutation(np.arange(X.size(0) ) )
    X_sample = X[perm[:min(n_sample, X.size(0) ) ], :]
    return X_sample

def get_matrix_spectrum(X):
    # linalg.eigh can give negative value on cencus regression dataset
    # So we use svd here and we have not seen numerical issue yet.
    # currently only works for symetric matrix
    # when using torch mm for X1X1, it can produce slight different values in 
    # the upper and lower parts, but tested to be within tolerance using
    # np.testing.assert_array_almost_equal
    # if not torch.equal(X, torch.transpose(X, 0, 1) ):
    #     raise Exception("Kernel matrix is not symetric!")
    S, U = np.linalg.eigh(X.cpu().numpy().astype(np.float64), UPLO='U')
    if np.min(S) <= 0:
        print("numpy eigh gives negative values, switch to use SVD")
        U, S, _ = np.linalg.svd(X.cpu().numpy().astype(np.float64) )
    return S 

def get_sample_kernel_metrics(X, kernel, kernel_approx, quantizer, n_sample):
    X_sample = sample_data(X, n_sample)
    kernel_mat = kernel.get_kernel_matrix(X, X)
    kernel_mat_approx = kernel_approx.get_kernel_matrix(X, X, quantizer, quantizer)
    # # need to use double for XXT if we want the torch equal to hold.
    # if not torch.equal(kernel_mat_approx, torch.transpose(kernel_mat_approx, 0, 1) ):
    #     raise Exception("Kernel matrix is not symetric!")
    error_matrix = kernel_mat_approx - kernel_mat
    F_norm_error = torch.sum(error_matrix**2)
    spectral_norm_error = np.max(np.abs(get_matrix_spectrum(error_matrix) ) )
    spectrum = get_matrix_spectrum(kernel_mat_approx)
    metric_dict = {"F_norm_error": float(F_norm_error),
                   "spectral_norm_error": float(spectral_norm_error) }
    return metric_dict, spectrum



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