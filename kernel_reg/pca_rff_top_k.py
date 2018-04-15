import numpy as np
import torch

import sys, os
sys.path.append("../utils")
from data_loader import load_census_data, load_census_data_part
from rff import GaussianKernel, RFF
from pca_rff import PCA_RFF
from bit_assignment import binary_search_bits_assignment
from kernel_regressor import Quantizer, QuantizerAutoScale


class PCA_RFFTopK(PCA_RFF):
  def __init__(self, n_feat, n_input_feat, kernel=None, rand_seed=1, mu=1.0):
    super(PCA_RFFTopK, self).__init__(n_feat, n_input_feat, kernel, rand_seed)
    self.mu = mu # the value to deside the scale
    self.mode = "train"
    
  def setup(self, input, n_top_comp, n_fp_feat_budget, percentile, residual_bit, bit_upperbound=32):
    self.n_top_comp = n_top_comp
    self.residual_bit = residual_bit 
    # percentile to determine dynamic range of residuals
    self.percentile = percentile
    rff = self.get_cos_feat(input)  
    if rff.size(0) < self.n_feat:
      raise Exception("number of samples should be large than the number of rff features for setup")    
    self.offset = rff.mean(0).unsqueeze(0) 
    rff_center = rff - self.offset.expand(rff.size(0), rff.size(1) )
    U, S, _ = np.linalg.svd(torch.transpose(rff_center, 0, 1).cpu().numpy().astype(np.float64), full_matrices=True)
    self.U_top_k = torch.DoubleTensor(U[:, :self.n_top_comp] )
    self.top_k_mem_budget = n_fp_feat_budget - self.n_feat * residual_bit / 32.0
    self.bit_assignment_top_k = \
      binary_search_bits_assignment(np.abs(S[:self.n_top_comp] ), 
      self.top_k_mem_budget, upper_bound=bit_upperbound)
    self.train_mode()

  def transform_cos_feat(self, rff):
    '''
    the rff comes in the shape of [n_sample, n_dim]
    '''
    rff_center = rff - self.offset.expand(rff.size(0), rff.size(1) )
    rff_top_k = torch.mm(rff_center, self.U_top_k)
    rff_residual = rff_center - torch.mm(rff_top_k, torch.transpose(self.U_top_k, 0, 1) )
    return rff_top_k, rff_residual
  
  def apply_quantization(self, rff_x, quantizer, bit_assignment):
    # for debug only
    fp_rff_x = rff_x.clone()

    # for top K quantization
    assert self.n_top_comp == len(bit_assignment)
    for i, nbits in enumerate(bit_assignment):
      if nbits == 0:
        rff_x[:, i] = 0.0
        continue
      if quantizer != None:
        # abs(eigen value) / sqrt(n_sample) is the std on the corresponding dimension
        min_val = None
        max_val = None
        quant = quantizer(nbits, min_val, max_val, rand_seed=self.rand_seed)
        #                 print("quantization 1 activated ", X1.shape)
        #                 print("quantizer 1 bits", quantizer.nbit)
        #                 print("quantizer 1 scale", quantizer.scale)
        #                 print torch.std(rff_x1[:, i] ), self.std[i]
        rff_x[:, i] = quant.quantize(rff_x[:, i], verbose=False)        
        # assert quantization is carried out properly
        if type(quant.min_val) is not float and type(quant.min_val) is not np.float64:
          assert np.abs( ( (rff_x[-1, i] - quant.min_val[-1, 0]) / quant.scale[-1, 0]) \
                - float(round( (rff_x[-1, i] - quant.min_val[-1, 0] ) / quant.scale[-1, 0], 0) ) ) <= 1e-6
        else:
          assert np.abs( ( (rff_x[-1, i] - quant.min_val) / quant.scale) \
                - float(round( (rff_x[-1, i] - quant.min_val) / quant.scale, 0) ) ) <= 1e-6
        # if is a auto scale quantizer, assert the max and min value matches the percentile
        if quantizer != None and isinstance(quant, QuantizerAutoScale):
          print("quantizer percentile", quantizer.percentile)
          test_val = rff_x.cpu().numpy()
          np.testing.assert_array_almost_equal(np.percentile(fp_rff_x.cpu().numpy(), q=quant.percentile, axis=0),
            np.min(test_val, axis=0) )
          np.testing.assert_array_almost_equal(np.percentile(fp_rff_x.cpu().numpy(), q=100.0-quanti.percentile, axis=0),
            np.max(test_val, axis=0) )
    return rff_x


  def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None):
    '''
    X1 shape is [n_sample, n_dim]
    quantizer1 and 2 are a quantizer class name
    '''
    rff_x1 = self.get_cos_feat(X1)
    rff_x1_top_k, rff_x1_residual = self.transform_cos_feat(rff_x1)
    rff_x2 = self.get_cos_feat(X2)
    rff_x2_top_k, rff_x2_residual = self.transform_cos_feat(rff_x2)
    
    rff_x1_top_k = self.apply_quantization(rff_x1_top_k, quantizer1, self.bit_assignment_top_k)
    rff_x2_top_k = self.apply_quantization(rff_x2_top_k, quantizer2, self.bit_assignment_top_k)
    rff_x1_residual = self.apply_quantization(rff_x1_residual, quantizer1, [self.residual_bit]*self.n_feat)
    rff_x2_residual = self.apply_quantization(rff_x2_residual, quantizer2, [self.residual_bit]*self.n_feat)    
    self.rff_x1 = torch.mm(rff_x1_top_k, torch.transpose(self.U_top_k, 0, 1) ) + rff_x1_residual + self.offset
    self.rff_x2 = torch.mm(rff_x2_top_k, torch.transpose(self.U_top_k, 0, 1) ) + rff_x2_residual + self.offset
    return torch.mm(self.rff_x1, torch.transpose(self.rff_x2, 0, 1) )


def pca_rff_top_k_fp_test():
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
  kernel = PCA_RFFTopK(1024, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_top_comp=1024, n_fp_feat_budget=1024, percentile=0.0, residual_bit=0, bit_upperbound=32)
  kernel_mat_fp_pca_rff = kernel.get_kernel_matrix(X_train, X_train)

  np.testing.assert_array_almost_equal(kernel_mat_fp_rff.cpu().numpy(), kernel_mat_fp_pca_rff.cpu().numpy(), decimal=6)
  print("pca rff top k fp test passed!")


if __name__ == "__main__":
  pca_rff_top_k_fp_test()

