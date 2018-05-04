import numpy as np
from scipy.optimize import minimize

# for numerical protection
EPS = 1e-20

def expected_loss(lam, U, S, Y, noise):
    m = float(Y.size)
    uty2 = ( (np.dot(U.T, Y.reshape(Y.size, 1) ) )**2).reshape(int(m))
    gamma = (S/(S + lam + EPS) ).reshape(int(m))
    return (1/m) * np.sum(((1.0-gamma)**2) * uty2) + (1/m)*noise**2 * np.sum(gamma**2) + noise**2
    # define U,S,Y,noise
    # f = lambda lam: expectedLoss(lam,U,S,Y,noise)
    # x0 = 10
    # res = minimize(f, x0, method='nelder-mead', options={'xtol': 1e-10, 'disp': True})
    # loss = f(res.x)
    # print(res.x)
    # print(loss)

def delta_approximation(K, K_tilde, lambda_=1e-3):
    """ Compute the smallest D such that (1 - D)(K + lambda_ I) <= K_tilde + lambda_ I <= (1 + D)(K + lambda_ I),
    where the inequalities are in semidefinite order.
    We won't check that the kernel matrices are positive semidefinite, so the
    result might be wrong if K and K_tilde aren't PSD.
    As in the proof of Lemma 8 of https://arxiv.org/pdf/1804.09893.pdf, if K + lambda_ I has
    the eigen-decomposition V Sigma^2 V.T, then
    D = ||Sigma^{-1} V.T K_tilde V Sigma^{-1} - Sigma^{-1} V.T K V Sigma^{-1}||
    Parameters:
        K: kernel matrix (PSD), n x n numpy array.
        K_tile: approximate kernel matrix (PSD), n x n numpy array.
        lambda_: real number.
        precision: precision of the returned D.
    Return:
        D: real number.
    """
    n, m = K.shape
    n_tilde, m_tilde = K_tilde.shape
    assert n == m and n_tilde == m_tilde, "Kernel matrix must be square"
    assert n == n_tilde, "K and K_tilde must have the same shape"
    assert np.allclose(K, K.T) and np.allclose(K_tilde, K_tilde.T), "Kernel matrix must be symmetric"
    # Compute eigen-decomposition of K + lambda_ I, of the form V @ np.diag(sigma) @ V.T
    sigma, V = np.linalg.eigh(K)
    assert np.all(sigma >= 0), "Kernel matrix K must be positive semidefinite"
    sigma += lambda_
    # Whitened K_tilde: np.diag(1 / np.sqrt(sigma)) @ V.T @ K_tilde @ V @ np.diag(1 / np.sqrt(sigma))
    K_tilde_whitened = V.T.dot(K_tilde.dot(V)) / np.sqrt(sigma) / np.sqrt(sigma)[:, np.newaxis]
    K_whitened = np.diag(1 - lambda_ / sigma)
    return np.linalg.norm(K_tilde_whitened - K_whitened, ord=2)


def delta_approximation_test():
    n = 5
    phi = np.random.rand(n, n)
    K = phi.T.dot(phi)
    K_tilde = 0.8 * K
    # phi_tilde = np.random.rand(n, n)
    # K_tilde = phi.T.dot(phi) + 0.5 * K
    lambda_ = 1
    D = delta_approximation(K, K_tilde, lambda_)
    # Either of these values should be very close to 0
    print(np.linalg.eigvalsh((1 + D) * (K + lambda_ * np.eye(n)) - (K_tilde + lambda_ * np.eye(n))).min())
    print(np.linalg.eigvalsh((K_tilde + lambda_ * np.eye(n)) - (1 - D) * (K + lambda_ * np.eye(n))).min())

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
