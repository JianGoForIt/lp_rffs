import numpy as np
import torch
import sys
sys.path.append("../utils")
from misc_utils import set_random_seed

class GaussianKernelSpec(object):
  def __init__(self, sigma):
    self.sigma = sigma

class GaussianKernel(object):
  def __init__(self, sigma):
    self.sigma = sigma
    self.dist_func = torch.nn.PairwiseDistance(p=2)
  
  def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None, dtype="float"):
    '''
    the input value has shape [n_sample, n_dim]
    quantizer is dummy here
    dtype only works for numpy input for X1 X2
    '''
    if isinstance(X1, np.ndarray) and isinstance(X2, np.ndarray):
      n_sample_X1 = X1.shape[0]
      norms_X1 = np.linalg.norm(X1, axis=1).reshape(n_sample_X1, 1)
      n_sample_X2 = X2.shape[0]
      norms_X2 = np.linalg.norm(X2, axis=1).reshape(n_sample_X2, 1)
      cross = np.dot(X1, X2.T)
      # print("using sigma ", self.sigma)
      kernel = np.exp(-0.5 / float(self.sigma)**2 \
        * (np.tile(norms_X1**2, (1, n_sample_X2) ) + np.tile( (norms_X2.T)**2, (n_sample_X1, 1) ) \
        -2 * cross) )
      if dtype == "float":
          return torch.Tensor(kernel).float()
      else:
          return torch.Tensor(kernel).double()
    else:
      ## to prevent memory explosion on GPU, we can do the following operations on CPU and move results
      ## back to GPU
      #is_cuda_tensor = X1.is_cuda      
      #if is_cuda_tensor and use_cpu_comp:
      #    X1 = X1.cpu()
      #    X2 = X2.cpu()
      norms_X1 = (X1**2).sum(1).view(-1, 1)
      norms_X2 = (X2**2).sum(1).view(-1, 1)
      norms_X1 = norms_X1.repeat(1, int(X2.size(0) ) )
      norms_X2 = torch.transpose(norms_X2.repeat(1, int(X1.size(0) ) ), 0, 1)
      cross = torch.mm(X1, torch.transpose(X2, 0, 1) )
      kernel = torch.exp(-0.5 / float(self.sigma)**2 * (norms_X1 + norms_X2 - 2* cross) )
      #if is_cuda_tensor and use_cpu_comp:
      #    return kernel.cuda()
      #else:
      #    return kernel
      return kernel

  def torch(self, cuda=False):
    '''
    adapt the interface to the model launching wrapper
    '''
    pass
  
  def cpu(self):
    '''
    adapt the interface when switch parameter of some kernels back to cpu mode
    '''
    pass


class RFF(object):
  def __init__(self, n_feat, n_input_feat, kernel=None, rand_seed=1):
    self.n_feat = n_feat  # number of rff features
    self.kernel = kernel
    self.n_input_feat = n_input_feat # dimension of the original input
    self.rand_seed = rand_seed
    self.get_gaussian_wb()

  def get_gaussian_wb(self):
    # print("using sigma ", 1.0/float(self.kernel.sigma), "using rand seed ", self.rand_seed)
    np.random.seed(self.rand_seed)
    self.w = np.random.normal(scale=1.0/float(self.kernel.sigma), 
      size=(self.n_feat, self.n_input_feat) )
    # print("using n rff features ", self.w.shape[0] )
    np.random.seed(self.rand_seed)
    self.b = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(self.n_feat, 1) )

  def torch(self, cuda=False):
    self.w = torch.DoubleTensor(self.w)
    self.b = torch.DoubleTensor(self.b)
    if cuda:
      self.w = self.w.cuda()
      self.b = self.b.cuda()

  def cpu(self):
    self.w = self.w.cpu()
    self.b = self.b.cpu()

  def get_cos_feat(self, input_val, dtype="double"):
    # input are original representaiton with the shape [n_sample, n_dim]
    if isinstance(self.kernel, GaussianKernel):
      if isinstance(input_val, np.ndarray):
        self.input = input_val.T
        self.feat = np.sqrt(2/float(self.n_feat) ) * np.cos(np.dot(self.w, self.input) + self.b)
        if dtype=="double":
          return torch.DoubleTensor(self.feat.T)
        else:
          return torch.FloatTensor(self.feat.T)
      else:
        self.input = torch.transpose(input_val, 0, 1)
        self.feat = float(np.sqrt(2/float(self.n_feat) ) ) * torch.cos(torch.mm(self.w, self.input) + self.b)
        return torch.transpose(self.feat, 0, 1)
    else:
      raise Exception("the kernel type is not supported yet")

  def get_sin_cos_feat(self, input_val):
    pass

  def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None, consistent_quant_seed=True):
    '''
    X1 shape is [n_sample, n_dim], if force_consistent_random_seed is True
    the quantization will use the same random seed for quantizing rff_x1 and rff_x2
    '''
    rff_x1 = self.get_cos_feat(X1)
    rff_x2 = self.get_cos_feat(X2)

    if consistent_quant_seed and (quantizer1 is not None) and (quantizer2 is not None):
      assert quantizer1.rand_seed == quantizer2.rand_seed, "quantizer random seed are different under consistent quant seed mode!"
    if quantizer1 != None:
      if consistent_quant_seed and list(rff_x1.size() ) == list(rff_x2.size() ):
        print("quantizing rff_x1 with random seed", quantizer1.rand_seed)
        set_random_seed(quantizer1.rand_seed)
      else:
        print("quantizing rff_x1 without fixed random seed")
      # print("quantization 1 activated ", X1.shape)
      # print("quantizer 1 bits", quantizer1.nbit)
      # print("quantizer 1 scale", quantizer1.scale)
      rff_x1 = quantizer1.quantize(rff_x1)
    if quantizer2 != None:
      if consistent_quant_seed:
        print("quantizing rff_x2 with random seed", quantizer2.rand_seed)
        set_random_seed(quantizer2.rand_seed)
      # print("quantization 2 activated ", X2.shape)
      # print("quantizer 2 bits", quantizer2.nbit)
      # print("quantizer 2 scale", quantizer2.scale)
      rff_x2 = quantizer2.quantize(rff_x2)
    self.rff_x1, self.rff_x2 = rff_x1, rff_x2
    return torch.mm(rff_x1, torch.transpose(rff_x2, 0, 1) )
    

def test_pytorch_gaussian_kernel():
  n_feat = 10
  input_val  = np.ones( [2, n_feat] )
  input_val[0, :] *= 1
  input_val[0, :] *= 2
  # get exact gaussian kernel
  kernel = GaussianKernel(sigma=2.0)
  kernel_mat = kernel.get_kernel_matrix(input_val, input_val)
  kernel_mat_torch = kernel.get_kernel_matrix(torch.Tensor(input_val), torch.Tensor(input_val) )
  np.testing.assert_array_almost_equal(kernel_mat.cpu().numpy(), kernel_mat_torch.cpu().numpy() )
  print("gaussian kernel pytorch version test passed!")


def test_rff_generation():
  n_feat = 10
  n_rff_feat = 1000000
  input_val  = np.ones( [2, n_feat] )
  input_val[0, :] *= 1
  input_val[0, :] *= 2
  # get exact gaussian kernel
  kernel = GaussianKernel(sigma=2.0)
  kernel_mat = kernel.get_kernel_matrix(input_val, input_val)
  # get RFF approximate kernel matrix
  rff = RFF(n_rff_feat, n_feat, kernel=kernel)
  rff.get_gaussian_wb()
  approx_kernel_mat = rff.get_kernel_matrix(input_val, input_val)
  np.testing.assert_array_almost_equal(approx_kernel_mat.cpu().numpy(), kernel_mat.cpu().numpy(), decimal=3)
  print("rff generation test passed!")

def test_rff_generation2():
  n_feat = 10
  n_rff_feat = 1000000
  input_val  = np.ones( [2, n_feat] )
  input_val[0, :] *= 1
  input_val[0, :] *= 2
  # get exact gaussian kernel
  kernel = GaussianKernel(sigma=2.0)
  # kernel_mat = kernel.get_kernel_matrix(input_val, input_val)
  # get RFF approximate kernel matrix
  rff = RFF(n_rff_feat, n_feat, kernel=kernel)
  rff.get_gaussian_wb()
  approx_kernel_mat = rff.get_kernel_matrix(input_val, input_val)
  rff.torch(cuda=False)
  approx_kernel_mat2 = rff.get_kernel_matrix(torch.DoubleTensor(input_val), torch.DoubleTensor(input_val) )
  np.testing.assert_array_almost_equal(approx_kernel_mat.cpu().numpy(), approx_kernel_mat2.cpu().numpy(), decimal=6)
  print("rff generation test 2 passed!")


if __name__ == "__main__":
  test_pytorch_gaussian_kernel()
  test_rff_generation()
  test_rff_generation2()


