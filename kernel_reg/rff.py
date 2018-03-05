import numpy as np
import torch

class GaussianKernelSpec(object):
  def __init__(self, sigma):
    self.sigma = sigma

class GaussianKernel(object):
  def __init__(self, sigma):
    self.sigma = sigma

  def get_kernel_matrix(self, X1, X2):
    '''
    the input value has shape [n_sample, n_dim]
    '''
    n_sample_X1 = X1.shape[0]
    n_sample_X2 = X2.shape[0]
    norms_X1 = np.linalg.norm(X1, axis=1).reshape(n_sample_X1, 1)
    norms_X2 = np.linalg.norm(X2, axis=1).reshape(n_sample_X2, 1)
    cross = np.dot(X1, X2.T)
    kernel = np.exp(-0.5 / float(self.sigma)**2 \
      * (np.tile(norms_X1**2, (1, n_sample_X2) ) + np.tile( (norms_X2.T)**2, (n_sample_X1, 1) ) \
      -2 * cross) )
    return torch.FloatTensor(kernel)


class RFF(object):
  def __init__(self, n_feat, n_input_feat, kernel=None):
    self.n_feat = n_feat  # number of rff features
    self.kernel = kernel
    self.n_input_feat = n_input_feat # dimension of the original input
    self.get_gaussian_wb()

  def get_gaussian_wb(self):
    self.w = np.random.normal(scale=1.0/float(self.kernel.sigma), 
      size=(self.n_feat, self.n_input_feat) )
    self.b = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(self.n_feat, 1) )

  def get_cos_feat(self, input_val):
    # input are original representaiton with the shape [n_sample, n_dim]
    self.input = input_val.T
    if isinstance(self.kernel, GaussianKernel):
      self.feat = np.sqrt(2/float(self.n_feat) ) * np.cos(np.dot(self.w, self.input) + self.b)
    else:
      raise Exception("the kernel type is not supported yet")
    return torch.FloatTensor(self.feat.T)

  def get_sin_cos_feat(self, input_val):
    pass

  def get_kernel_matrix(self, X1, X2):
    '''
    X1 shape is [n_sample, n_dim]
    '''
    rff_x1 = self.get_cos_feat(X1)
    rff_x2 = self.get_cos_feat(X2)
    self.rff_x1, self.rff_x2 = rff_x1, rff_x2
    return torch.mm(rff_x1, torch.transpose(rff_x2, 0, 1) )


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

if __name__ == "__main__":
  test_rff_generation()


