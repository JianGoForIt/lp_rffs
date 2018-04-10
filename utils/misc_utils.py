import numpy as np
from scipy.optimize import minimize

def expected_loss(lam, U, S, Y, noise):
    m = np.ndarray.size(Y)
    uty2 = (np.dot(U.T, Y.reshape(Y.size, 1) ) )**2
    gamma = S/(S + lam)
    return (1/m) * np.sum(((1-gamma)**2) * uty2) + (1/m)*noise**2 * np.sum(gamma**2);
    # define U,S,Y,noise
    # f = lambda lam: expectedLoss(lam,U,S,Y,noise)
    # x0 = 10
    # res = minimize(f, x0, method='nelder-mead', options={'xtol': 1e-10, 'disp': True})
    # loss = f(res.x)
    # print(res.x)
    # print(loss)


class Args(object):
    def __init__(self, n_fp_rff, n_bit, 
                 exact_kernel, reg_lambda, 
                 sigma, random_seed, data_path,
                 do_fp, test_var_reduce=False):
        self.n_fp_rff = n_fp_rff
        self.n_bit = n_bit
        self.exact_kernel = exact_kernel
        self.reg_lambda = reg_lambda
        self.sigma = sigma
        self.random_seed = random_seed
        self.data_path = data_path
        self.do_fp = do_fp
        self.test_var_reduce = test_var_reduce