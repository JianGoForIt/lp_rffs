import numpy as np
import torch
from rff import GaussianKernel, RFF
from time import time
import math

class Quantizer(object):
  def __init__(self, nbit, min_val, max_val, scale=None, rand_seed=1, use_cuda=False):
    self.nbit = nbit
    self.min_val = min_val
    self.max_val = max_val
    if scale == None:
      self.scale = (max_val - min_val) / float(2**self.nbit - 1)
    self.rand_seed = rand_seed
    self.use_cuda = use_cuda

  def quantize_random(self, value, verbose=True, test=False):
    bound = math.pow(2.0, self.nbit) - 1
    min_val = 0.0
    max_val = bound
    # Generate tensor of random values from [0,1]
    # np.random.seed(self.rand_seed)
    if self.use_cuda:
      if test:
        adj_val = torch.cuda.Tensor(np.random.uniform(size=list(value.size() ) ) ).type(value.type() )
      else:
        adj_val = torch.cuda.Tensor(value.size()).type(value.type()).uniform_()
    else:
      if test:
        adj_val = torch.Tensor(np.random.uniform(size=list(value.size() ) ) ).type(value.type() )
      else:
        adj_val = torch.Tensor(value.size()).type(value.type()).uniform_()
    rounded = (value - self.min_val).div_(self.scale).add_(adj_val).floor_()
    clipped_value = rounded.clamp_(min_val, max_val)
    clipped_value *= self.scale 
    quant_val = clipped_value + self.min_val
    return quant_val

  def quantize_random_old(self, value, verbose=True):
    floor_val = self.min_val + torch.floor( (value - self.min_val) / self.scale) * self.scale
    ceil_val = self.min_val + torch.ceil( (value - self.min_val) / self.scale) * self.scale
    # print("test in the middle ", torch.min(floor_val), torch.max(ceil_val), self.min_val, self.max_val)
    # exit(0)
    floor_prob = (ceil_val - value) / self.scale
    ceil_prob = (value - floor_val) / self.scale
    # sanity check
    # np.testing.assert_array_almost_equal(floor_prob.cpu().numpy(), 
    #   1 - ceil_prob.cpu().numpy(), decimal=6)
    # if verbose:
    #   print("quantizer using random seed", self.rand_seed)
    np.random.seed(self.rand_seed)
    sample = torch.DoubleTensor(np.random.uniform(size=list(value.size() ) ) )
    # quant_val = floor_val * (sample < floor_prob).float() \
    #   + ceil_val * (sample >= floor_prob).float()
    quant_val = floor_val * (sample < floor_prob).double() \
      + ceil_val * (sample >= floor_prob).double()
    return quant_val

  def quantize(self, value, verbose=True, test=False):
    # TODO update if we have other quantization schemes
    value = torch.clamp(value, self.min_val, self.max_val)
    return self.quantize_random(value, verbose, test)

  def quantize_old(self, value, verbose=True):
    # TODO update if we have other quantization schemes
    value = torch.clamp(value, self.min_val, self.max_val)
    return self.quantize_random_old(value, verbose)


class QuantizerAutoScale(Quantizer):
  def __init__(self, nbit, min_val=0.0, max_val=0.0, scale=None, rand_seed=1, percentile=0):
    '''
    min_val and max_val are dummy here
    '''
    super(QuantizerAutoScale, self).__init__(nbit, min_val, max_val, scale=scale, rand_seed=rand_seed)
    self.percentile = percentile

  def quantize(self, value, verbose=True, test=True):
    '''
    values come in the shape of [n_sample, n_feature]
    '''
    # percentile based quantization
    value_np = value.cpu().numpy()
    min_val = np.percentile(value_np, q=self.percentile, axis=0)
    max_val = np.percentile(value_np, q=(100.0 - self.percentile), axis=0)
    value_np = np.clip(value_np, min_val, max_val)
    scale = (max_val - min_val) / float(2**self.nbit - 1)
    self.scale = torch.DoubleTensor(np.tile(scale.reshape(1, scale.size), (value.size(0), 1) ) )
    self.min_val = torch.DoubleTensor(np.tile(min_val.reshape(1, min_val.size), (value.size(0), 1) ) )
    self.max_val = torch.DoubleTensor(np.tile(max_val.reshape(1, max_val.size), (value.size(0), 1) ) )
    if len(value_np.shape) == 1:
      value_np = value_np.reshape(value_np.size, 1)
    value = torch.DoubleTensor(value_np)
    # print self.scale, self.min_val, self.max_val, np.min(value_np, axis=0), np.max(value_np, axis=0)
    return self.quantize_random(value, verbose, test)



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
    self.kernel_mat = self.kernel.get_kernel_matrix(X_train, X_train, quantizer, quantizer)
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
    self.alpha = torch.DoubleTensor( \
      np.dot(np.linalg.inv( (self.kernel_mat + self.reg_lambda * torch.eye(n_sample).double() ).cpu().numpy() ), Y_train) )
    # self.alpha = torch.mm(torch.inverse(self.kernel_mat + self.reg_lambda * torch.eye(n_sample) ), 
    #   torch.DoubleTensor(Y_train) )

  def get_train_error(self):
    prediction = torch.mm(self.kernel_mat, self.alpha)
    error = prediction - torch.DoubleTensor(self.Y_train)
    return torch.mean(error**2)

  def predict(self, X_test, quantizer_train=None, quantizer_test=None):
    # quantizer 1 for test data, quantizer 2 for train data
    self.X_test = X_test
    self.kernel_mat_pred = \
      self.kernel.get_kernel_matrix(self.X_test, self.X_train, quantizer_test, quantizer_train)
    self.prediction = torch.mm(self.kernel_mat_pred, self.alpha)
    return self.prediction.clone()

  def get_test_error(self, Y_test):
    # should only be called right after the predict function
    self.Y_test = Y_test
    error = self.prediction - torch.DoubleTensor(self.Y_test)
    return torch.mean(error**2)


def test_random_quantizer():
  quantizer = Quantizer(nbit=15, min_val=-2**14+1, max_val=2**14)

  # test lower bound
  lower = -2**14+1.0
  shift = 1/3.0
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value, test=True)
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
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value, test=True)
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
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value, test=True)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 0.95 and ratio < 1.05

  print("quantizer test passed!")


def test_random_quantizer_fast_impl():
  # this only works when use numpy setted seed in new fast random quantize implementation
  quantizer = Quantizer(nbit=15, min_val=-2**14+1, max_val=2**14)
  # test middle values
  lower = 0.0
  shift = 0.5
  # value = np.ones( (1000, 1000) ) * (lower + shift)
  value = np.random.uniform((1000, 1000)) * 2**14
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value, test=True)
  quant_val_old = quantizer.quantize_old(value)
  quant_val = quant_val.cpu().numpy()
  quant_val_old = quant_val_old.cpu().numpy()
  np.testing.assert_array_almost_equal(quant_val, quant_val_old, decimal=9)
  print("fast impl quantizer test passed!")


def test_auto_scale_random_quantizer():
  quantizer = Quantizer(nbit=15, min_val=-1.0, max_val=1.0, rand_seed=1)
  quantizer_auto = QuantizerAutoScale(nbit=15, percentile=0, rand_seed=quantizer.rand_seed)
  # test auto scale quantizer produces same output as base quantizer
  value = np.random.uniform(low=-1.0, high=1.0, size=(1000, 500) )
  value[0, :] = -1.0  # make the two quanizer has the same scale
  value[-1, :] = 1.0
  value = torch.DoubleTensor(value)
  np.random.seed(quantizer.rand_seed)
  quant_val = quantizer.quantize(value, test=True)
  quant_val = quant_val.cpu().numpy()
  np.random.seed(quantizer.rand_seed)
  auto_quant_val = quantizer_auto.quantize(value, test=True)
  auto_quant_val = auto_quant_val.cpu().numpy()
  np.testing.assert_array_almost_equal(auto_quant_val, quant_val, decimal=7)

  quantizer_auto = QuantizerAutoScale(nbit=15, percentile=10, rand_seed=quantizer.rand_seed)
  # test auto scale quantizer produces same output as base quantizer
  value = np.random.uniform(low=-1.0, high=1.0, size=(1000, 500) )
  value[0, :] = -1.0  # make the two quanizer has the same scale
  value[-1, :] = 1.0
  value = torch.DoubleTensor(value)
  # np.random.seed(quantizer.rand_seed)
  auto_quant_val = quantizer_auto.quantize(value, test=True)
  auto_quant_val = auto_quant_val.cpu().numpy()
  np.testing.assert_array_almost_equal(np.max(auto_quant_val, axis=0), 
    np.percentile(value.cpu().numpy(), q=90, axis=0) )
  np.testing.assert_array_almost_equal(np.min(auto_quant_val, axis=0), 
    np.percentile(value.cpu().numpy(), q=10, axis=0) )
  print("auto scale quantizer test passed!")


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
  val = torch.inverse( (regressor.reg_lambda * torch.eye(n_rff_feat).double() \
    + torch.mm(torch.transpose(kernel.rff_x1, 0, 1), kernel.rff_x1) ) )
  val = torch.mm(val, torch.transpose(kernel.rff_x2, 0, 1) )
  w2 = torch.mm(val, torch.DoubleTensor(Y_train) )
  np.testing.assert_array_almost_equal(w1.cpu().numpy(), w2.cpu().numpy() )
  # print(w1.cpu().numpy().ravel()[-10:-1], w2.cpu().numpy().ravel()[-10:-1] )
  print("kernel ridge regression test2 passed!")


if __name__ == "__main__":
  test_random_quantizer_fast_impl()
  test_random_quantizer()
  test_auto_scale_random_quantizer()
  test_kernel_ridge_regression1()
  test_kernel_ridge_regression2()



