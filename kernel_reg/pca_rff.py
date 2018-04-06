import numpy as np
import torch

import sys, os
sys.path.append("../utils")
from data_loader import load_census_data, load_census_data_part
from rff import GaussianKernel, RFF
from bit_assignment import binary_search_bits_assignment
from kernel_regressor import Quantizer


class PCA_RFF(RFF):
  def __init__(self, n_feat, n_input_feat, kernel=None, rand_seed=1, mu=1.0):
    super(PCA_RFF, self).__init__(n_feat, n_input_feat, kernel, rand_seed)
    self.mu = mu # the value to deside the scale
    self.mode = "train"

  def train_mode(self):
    self.mode = "train"

  def test_mode(self):
    self.mode = "test"
  
  def setup(self, input, n_fp_feat_budget, bits_upperbound=32):
    '''
    the input comes in the shape of [n_sample, n_dim]
    n_fp_feat_budget the budget in the unit of fp feat
    bits_upperbound indicates whether we have a maximum bit limit for each number
    '''
    rff = self.get_cos_feat(input)  
    if rff.size(0) < self.n_feat:
      raise Exception("number of samples should be large than the number of rff features for setup")    
    self.offset = rff.mean(0).unsqueeze(0) 
    rff_center = rff - self.offset.expand(rff.size(0), rff.size(1) )
    U, S, _ = np.linalg.svd(torch.transpose(rff_center, 0, 1).cpu().numpy().astype(np.float64), full_matrices=True)
    # ZU gives the transformed rff with rff of the shape [n_sample, n_dim]
    self.U = torch.DoubleTensor(U)
    self.offset_rot = torch.mm(self.offset, self.U)
    # unbiased estimation of std on each eigen direction 
    self.std = np.abs(S) / np.sqrt(rff.size(0) - 1.0)
    rff_center_rot = torch.mm(rff_center, self.U)
    # self test, assert the rff transformation is correctedly carried out
    test_feat = torch.mm(rff_center_rot, torch.transpose(self.U, 0, 1) ) + self.offset.expand(rff.size(0), rff.size(1) )
    np.testing.assert_array_almost_equal(rff.cpu().numpy(), test_feat.cpu().numpy(), decimal=6)   
    # assign variant bits
    self.bit_assignment = binary_search_bits_assignment(self.mu * self.std, n_fp_feat_budget, upper_bound=bits_upperbound)
    self.train_mode()

  def transform_cos_feat(self, rff):
    '''
    the rff comes in the shape of [n_sample, n_dim]
    '''
    rff_center = rff - self.offset.expand(rff.size(0), rff.size(1) )
    rff_center_rot = torch.mm(rff_center, self.U)
    return rff_center_rot

  def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None):
    '''
    X1 shape is [n_sample, n_dim]
    quantizer1 and 2 are a quantizer class name
    '''
    rff_x1 = self.get_cos_feat(X1)
    rff_x1 = self.transform_cos_feat(rff_x1)
    rff_x2 = self.get_cos_feat(X2)
    rff_x2 = self.transform_cos_feat(rff_x2)
    for i, nbits in enumerate(self.bit_assignment):
      if nbits == 0:
        rff_x1[:, i] = 0.0
        rff_x2[:, i] = 0.0
        continue
      if quantizer1 != None:
        # abs(eigen value) / sqrt(n_sample) is the std on the corresponding dimension
        min_val = -self.std[i] * self.mu
        max_val = self.std[i] * self.mu
        quantizer = quantizer1(nbits, min_val, max_val, rand_seed=self.rand_seed)
#                 print("quantization 1 activated ", X1.shape)
#                 print("quantizer 1 bits", quantizer.nbit)
#                 print("quantizer 1 scale", quantizer.scale)
#                 print torch.std(rff_x1[:, i] ), self.std[i]
        if self.mode == "train":
          np.testing.assert_array_almost_equal(torch.std(rff_x1[:, i] ) / self.std[i], 1.0, decimal=6)
        rff_x1[:, i] = quantizer.quantize(rff_x1[:, i], verbose=False)
        # print torch.min(rff_x1[:, i] - quantizer.min_val) / quantizer.scale, torch.max(rff_x1[:, i] - quantizer.min_val) / quantizer.scale
        assert np.abs( ( (rff_x1[-1, i] - quantizer.min_val) / quantizer.scale) \
                      - float(round( (rff_x1[-1, i] - quantizer.min_val) / quantizer.scale, 0) ) ) <= 1e-6
      if quantizer2 != None:
        min_val = -self.std[i] * self.mu
        max_val = self.std[i] * self.mu
        quantizer = quantizer2(nbits, min_val, max_val, rand_seed=self.rand_seed)
#                 print("quantization 2 activated ", X2.shape)
#                 print("quantizer 2 bits", quantizer.nbit)
#                 print("quantizer 2 scale", quantizer.scale)
#                 print torch.std(rff_x2[:, i] ), self.std[i]
        if self.mode == "train":
          np.testing.assert_array_almost_equal(torch.std(rff_x2[:, i] ) / self.std[i], 1.0, decimal=6)
        rff_x2[:, i] = quantizer.quantize(rff_x2[:, i], verbose=False)
        # print torch.min( (rff_x2[:, i] - quantizer.min_val) / quantizer.scale), torch.max((rff_x2[:, i] - quantizer.min_val) / quantizer.scale)
        assert np.abs( ( (rff_x2[-1, i] - quantizer.min_val) / quantizer.scale) \
                      - float(round( (rff_x2[-1, i] - quantizer.min_val) / quantizer.scale, 0) ) ) <= 1e-6
    self.rff_x1, self.rff_x2 = rff_x1 + self.offset_rot, rff_x2 + self.offset_rot
    return torch.mm(self.rff_x1, torch.transpose(self.rff_x2, 0, 1) )


def pca_rff_fp_test():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]

  # get a full precision rff kernel
  kernel = GaussianKernel(sigma=sigma)
  kernel = RFF(1024, n_input_feat, kernel, rand_seed=1)
  kernel_mat_fp_rff = kernel.get_kernel_matrix(X_train, X_train)

  # get a full precision PCA RFF
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFF(1024, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_fp_feat_budget=1024)
  kernel_mat_fp_pca_rff = kernel.get_kernel_matrix(X_train, X_train)

  np.testing.assert_array_almost_equal(kernel_mat_fp_rff.cpu().numpy(), kernel_mat_fp_pca_rff.cpu().numpy(), decimal=6)
  # print kernel_mat_fp_rff
  # print kernel_mat_fp_pca_rff
  # print kernel.offset
  # print kernel.S
  print("pca rff fp test passed!")
    
def pca_rff_32bit_test():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]
  quantizer = Quantizer
  # get a full precision rff kernel
  kernel = GaussianKernel(sigma=sigma)
  kernel = RFF(1024, n_input_feat, kernel, rand_seed=1)
  kernel_mat_fp_rff = kernel.get_kernel_matrix(X_train, X_train)

  # get a 32bit precision PCA RFF
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFF(1024, n_input_feat, kernel, rand_seed=1, mu=100.0)
  kernel.setup(X_train, n_fp_feat_budget=1024)
  kernel_mat_fp_pca_rff = kernel.get_kernel_matrix(X_train, X_train, quantizer, quantizer)

  np.testing.assert_array_almost_equal(kernel_mat_fp_rff.cpu().numpy(), kernel_mat_fp_pca_rff.cpu().numpy(), decimal=6)
  print("pca rff 32 test passed!")


if __name__ == "__main__":
  pca_rff_fp_test()
  pca_rff_32bit_test()

