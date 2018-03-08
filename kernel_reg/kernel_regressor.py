import numpy as np
import torch
from rff import GaussianKernel, RFF
from time import time

class Quantizer(object):
  def __init__(self, nbit, min_val, max_val, scale=None, rand_seed=1):
    self.nbit = nbit
    self.min_val = min_val
    self.max_val = max_val
    if scale == None:
      self.scale = (max_val - min_val) / float(2**self.nbit - 1)
    self.rand_seed = rand_seed

  def quantize_random(self, value):
    value = torch.clamp(value, self.min_val, self.max_val)
    floor_val = self.min_val + torch.floor( (value - self.min_val) / self.scale) * self.scale
    ceil_val = self.min_val + torch.ceil( (value - self.min_val) / self.scale) * self.scale
    # print("test in the middle ", torch.min(floor_val), torch.max(ceil_val), self.min_val, self.max_val)
    # exit(0)
    floor_prob = (ceil_val - value) / self.scale
    ceil_prob = (value - floor_val) / self.scale
    # sanity check
    # np.testing.assert_array_almost_equal(floor_prob.cpu().numpy(), 
    #   1 - ceil_prob.cpu().numpy(), decimal=6)
    print("quantizer using random seed", self.rand_seed)
    np.random.seed(self.rand_seed)
    sample = torch.FloatTensor(np.random.uniform(size=list(value.size() ) ) )
    quant_val = floor_val * (sample < floor_prob).float() \
      + ceil_val * (sample >= floor_prob).float()
    return quant_val

  def quantize(self, value):
    # TODO update if we have other quantization schemes
    return self.quantize_random(value)


class KernelRidgeRegression(object):
  def __init__(self, kernel, reg_lambda):
    '''
    reg_lambda is the strength of the regression regularizor
    kernel matrix is a Pytorch Tensor
    '''
    # self.kernel_mat = kernel_mat
    self.reg_lambda = reg_lambda
    self.kernel = kernel

  def fit(self, X_train=None, Y_train=None, kernel_mat=None, quantizer=None):
    self.X_train, self.Y_train = X_train, Y_train
    self.kernel_mat = self.kernel.get_kernel_matrix(X_train, X_train, quantizer)
    n_sample = self.kernel_mat.size(0)

    # # DEBUG
    # print("start timing ", self.kernel_mat.size() )
    # test = (self.kernel_mat + self.reg_lambda * torch.eye(n_sample) )
    # test = test.cpu().numpy()
    # start_time = time()
    # np.linalg.inv(test)
    # end_time = time()
    # print("numpy inverse time ", end_time - start_time)

    # # test = (self.kernel_mat + self.reg_lambda * torch.eye(n_sample) )[0:5000, 0:5000]
    # # start_time = time()
    # # torch.inverse(test)
    # # end_time = time()
    # # print("pytorch inverse time ", end_time - start_time)
    # exit(0)
    
    # pytorch is super slow in inverse, so we finish this operation in numpy
    print("using regularior strength ", self.reg_lambda)
    self.alpha = torch.FloatTensor( \
      np.dot(np.linalg.inv( (self.kernel_mat + self.reg_lambda * torch.eye(n_sample) ).cpu().numpy() ), Y_train) )
    # self.alpha = torch.mm(torch.inverse(self.kernel_mat + self.reg_lambda * torch.eye(n_sample) ), 
    #   torch.FloatTensor(Y_train) )

  def get_train_error(self):
    prediction = torch.mm(self.kernel_mat, self.alpha)
    error = prediction - torch.FloatTensor(self.Y_train)
    return torch.mean(error**2)

  def predict(self, X_test, quantizer=None):
    self.X_test = X_test
    self.kernel_mat_pred = \
      self.kernel.get_kernel_matrix(self.X_test, self.X_train, quantizer)
    self.prediction = torch.mm(self.kernel_mat_pred, self.alpha)
    return self.prediction.clone()

  def get_test_error(self, Y_test):
    # should only be called right after the predict function
    self.Y_test = Y_test
    error = self.prediction - torch.FloatTensor(self.Y_test)
    return torch.mean(error**2)


def test_random_quantizer():
  quantizer = Quantizer(nbit=15, min_val=-2**14+1, max_val=2**14)

  # test lower bound
  lower = -2**14+1.0
  shift = 1/3.0
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.FloatTensor(value)
  quant_val = quantizer.quantize_random(value)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 1.95 and ratio < 2.05

  # test upper bound
  lower = 2**14-1.0
  shift = 2/3.0
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.FloatTensor(value)
  quant_val = quantizer.quantize_random(value)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 0.45 and ratio < 0.55

  # test middle values
  lower = 0.0
  shift = 0.5
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.FloatTensor(value)
  quant_val = quantizer.quantize_random(value)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 0.95 and ratio < 1.05

  print("quantizer test passed!")


def test_kernel_ridge_regression1():
  '''
  We test the linear kernel case and gaussian kernel case
  '''
  n_feat = 10
  n_rff_feat = 1000000
  X_train  = np.ones( [2, n_feat] )
  X_train[0, :] *= 1
  X_train[0, :] *= 2
  Y_train = np.ones( [2, 1] )
  kernel = GaussianKernel(sigma=2.0)
  kernel = RFF(n_rff_feat, n_feat, kernel)
  reg_lambda = 1.0
  regressor = KernelRidgeRegression(kernel, reg_lambda=reg_lambda)
  regressor.fit(X_train, Y_train)

  # if test data equals traing data, it should the same L2 error
  X_test = np.copy(X_train)
  Y_test = np.copy(Y_train)
  test_pred = regressor.predict(X_test)
  train_error = regressor.get_train_error()
  test_error = regressor.get_test_error(Y_test)
  assert np.abs(train_error - test_error) < 1e-6

  # if test data is different from traing data, L2 error for train and test should be different
  X_test = np.copy(X_train) * 2
  Y_test = np.copy(Y_train)
  test_pred = regressor.predict(X_test)
  train_error = regressor.get_train_error()
  test_error = regressor.get_test_error(Y_test)
  assert np.abs(train_error - test_error) >= 1e-3

  X_test = np.copy(X_train)
  Y_test = np.copy(Y_train) * 2
  test_pred = regressor.predict(X_test)
  train_error = regressor.get_train_error()
  test_error = regressor.get_test_error(Y_test)
  assert np.abs(train_error - test_error) >= 1e-3

  print("kernel ridge regression test1 passed!")


def test_kernel_ridge_regression2():
  '''
  We test the linear kernel case and gaussian kernel case
  '''
  n_feat = 10
  n_rff_feat = 1000
  X_train  = np.ones( [2, n_feat] )
  X_train[0, :] *= 1
  X_train[0, :] *= 2
  Y_train = np.ones( [2, 1] )
  kernel = GaussianKernel(sigma=2.0)
  kernel = RFF(n_rff_feat, n_feat, kernel)
  reg_lambda = 1.0
  regressor = KernelRidgeRegression(kernel, reg_lambda=reg_lambda)
  regressor.fit(X_train, Y_train)

  # compare the two ways of calculating feature weights as sanity check
  # feature weight using the approach inside KernelRidgeRegression
  kernel.get_kernel_matrix(X_train, X_train)
  # print kernel.rff_x2.size(), regressor.alpha.size()
  w1 = torch.mm(torch.transpose(kernel.rff_x2, 0, 1), regressor.alpha)
  # print w1.size()
  # feature weight using alternative way of calculation
  val = torch.inverse( (regressor.reg_lambda * torch.eye(n_rff_feat) \
    + torch.mm(torch.transpose(kernel.rff_x1, 0, 1), kernel.rff_x1) ) )
  val = torch.mm(val, torch.transpose(kernel.rff_x2, 0, 1) )
  w2 = torch.mm(val, torch.FloatTensor(Y_train) )
  np.testing.assert_array_almost_equal(w1.cpu().numpy(), w2.cpu().numpy() )
  # print(w1.cpu().numpy().ravel()[-10:-1], w2.cpu().numpy().ravel()[-10:-1] )
  print("kernel ridge regression test2 passed!")


if __name__ == "__main__":
  test_random_quantizer()
  test_kernel_ridge_regression1()
  test_kernel_ridge_regression2()



