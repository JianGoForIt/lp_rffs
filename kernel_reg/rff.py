import numpy as np
import torch

class GaussianKernelSpec(object):
  def __init__(self, sigma):
    self.sigma = sigma

class GaussianKernel(object):
  def __init__(self, sigma):
    self.sigma = sigma

  def get_kernel_matrix(self, input_val):
    '''
    the input value has shape [n_sample, n_dim]
    '''
    n_sample = input_val.shape[0]
    norms = np.linalg.norm(input_val, axis=1).reshape(n_sample, 1)
    cross = np.dot(input_val, input_val.T)
    kernel = np.exp(-0.5 / float(self.sigma)**2 \
      * (np.tile(norms, (1, n_sample) ) + np.tile(norms.T, (n_sample, 1) ) \
      -2 * cross) )
    return torch.FloatTensor(kernel)


# TODO additioanl sanity check with exact gaussian kernel
class RFF(object):
  def __init__(self, n_feat, kernel=None):
    self.n_feat = n_feat
    self.kernel = kernel

  def get_gaussian_w(self):
    self.w = np.random.normal(scale=1.0/float(self.kernel.sigma), 
      size=(self.n_feat, self.input.shape[0] ) )

  def get_cos_feat(self, input_val):
    # input are original representaiton with the shape [n_dim, n_sample]
    self.input = input_val.T
    if isinstance(self.kernel, GaussianKernel):
      self.get_gaussian_w()
      self.b = np.random.uniform(low=0.0, high = 2.0 * np.pi, size=(self.n_feat, 1) )
      self.feat = np.sqrt(2/float(self.n_feat) ) * np.cos(np.dot(self.w, self.input) + self.b)
    else:
      raise Exception("the kernel type is not supported yet")
    return torch.FloatTensor(self.feat.T)

  def get_sin_cos_feat(self, input_val):
    pass


def test_rff_generation():
  n_feat = 10
  n_rff_feat = 1000000
  input_val  = np.ones( [2, n_feat] )
  input_val[0, :] *= 1
  input_val[0, :] *= 2
  # get exact gaussian kernel
  kernel = GaussianKernel(sigma=2.0)
  # get RFF approximate kernel matrix
  kernel_mat = kernel.get_kernel_matrix(input_val)
  rff = RFF(n_rff_feat, kernel=kernel)
  kernel_feat = rff.get_cos_feat(input_val)
  approx_kernel_mat = torch.mm(kernel_feat, torch.transpose(kernel_feat, 0, 1) )
  np.testing.assert_array_almost_equal(approx_kernel_mat.cpu().numpy(), kernel_mat.cpu().numpy() )
  print("rff generation test passed!")

if __name__ == "__main__":
  test_rff_generation()


