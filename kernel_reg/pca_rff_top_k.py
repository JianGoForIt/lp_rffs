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
    
  def setup(self, input, n_top_comp, n_fp_feat_budget, residual_bit, bits_upperbound=32):
    self.n_top_comp = n_top_comp
    self.residual_bit = residual_bit 
    rff = self.get_cos_feat(input)  
    # if rff.size(0) < self.n_feat:
    #  raise Exception("number of samples should be large than the number of rff features for setup")    
    self.offset = rff.mean(0).unsqueeze(0) 
    rff_center = rff - self.offset.expand(rff.size(0), rff.size(1) )
    U, S, _ = np.linalg.svd(torch.transpose(rff_center, 0, 1).cpu().numpy().astype(np.float64), full_matrices=True)
    self.U_top_k = torch.DoubleTensor(U[:, :self.n_top_comp] )
    self.top_k_mem_budget = n_fp_feat_budget - self.n_feat * residual_bit / 32.0
    if self.top_k_mem_budget <= 0:
      raise Exception("No memory left for top components!")
    self.bit_assignment_top_k = \
      binary_search_bits_assignment(np.abs(S[:self.n_top_comp] ), 
      self.top_k_mem_budget, upper_bound=bits_upperbound)
    print("budget for top K", self.top_k_mem_budget, "top K", len(self.bit_assignment_top_k), "total rff", rff.size(1) )
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
    # note can only use auto scale quantizer
    # for debug only
    fp_rff_x = rff_x.clone()
    # for top K quantization
    # assert self.n_top_comp == len(bit_assignment)
    # print("inside check ", bit_assignment)
    for i, nbits in enumerate(bit_assignment):
      if nbits == 0:
        rff_x[:, i] = 0.0
        # for debug
        fp_rff_x[:, i] = 0.0
        continue
      if quantizer != None:
        # abs(eigen value) / sqrt(n_sample) is the std on the corresponding dimension
        # set dummy min max value for auto scale quantizers
        min_val = 0.0
        max_val = 0.0
        quant = quantizer(nbits, min_val, max_val, rand_seed=self.rand_seed)
        #                 print("quantization 1 activated ", X1.shape)
        #                 print("quantizer 1 bits", quantizer.nbit)
        #                 print("quantizer 1 scale", quantizer.scale)
        #                 print torch.std(rff_x1[:, i] ), self.std[i]
        # if i == 1:
        #   print("inside quant i th ", torch.min(rff_x[:, i]), torch.max(rff_x[:, i]) )
        rff_x[:, i] = quant.quantize(rff_x[:, i], verbose=False)
        # if i == 1:
        #   print("inside quant i th after ", torch.min(rff_x[:, i]), torch.max(rff_x[:, i]) )        
        # assert quantization is carried out properly
        if type(quant.min_val) is not float and type(quant.min_val) is not np.float64:
          assert np.abs( ( (rff_x[-1, i] - quant.min_val[-1, 0]) / quant.scale[-1, 0]) \
                - float(round( (rff_x[-1, i] - quant.min_val[-1, 0] ) / quant.scale[-1, 0], 0) ) ) <= 1e-6
        else:
          assert np.abs( ( (rff_x[-1, i] - quant.min_val) / quant.scale) \
                - float(round( (rff_x[-1, i] - quant.min_val) / quant.scale, 0) ) ) <= 1e-6
        # if is a auto scale quantizer, assert the max and min value matches the percentile
      if (quantizer != None) and isinstance(quant, QuantizerAutoScale):
        test_val = rff_x.cpu().numpy()
        np.testing.assert_array_almost_equal(np.percentile(fp_rff_x[:, i].cpu().numpy(), q=quant.percentile, axis=0),
          np.min(test_val[:, i], axis=0) )
        np.testing.assert_array_almost_equal(np.percentile(fp_rff_x[:, i].cpu().numpy(), q=100.0-quant.percentile, axis=0),
          np.max(test_val[:, i], axis=0) )
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
    self.rff_x1_residual_raw, self.rff_x2_residual_raw = rff_x1_residual.clone(), rff_x2_residual.clone()
    
    np.random.seed(self.rand_seed)
    rff_x1_top_k = self.apply_quantization(rff_x1_top_k, quantizer1, self.bit_assignment_top_k)
    # make the quantization noise independent
    np.random.seed(self.rand_seed + 1)
    rff_x1_residual = self.apply_quantization(rff_x1_residual, quantizer1, [self.residual_bit]*self.n_feat)
    np.random.seed(self.rand_seed)
    rff_x2_top_k = self.apply_quantization(rff_x2_top_k, quantizer2, self.bit_assignment_top_k)
    np.random.seed(self.rand_seed + 1)
    rff_x2_residual = self.apply_quantization(rff_x2_residual, quantizer2, [self.residual_bit]*self.n_feat)    
    self.rff_x1_top_k, self.rff_x2_top_k = rff_x1_top_k, rff_x2_top_k
    self.rff_x1_residual, self.rff_x2_residual = rff_x1_residual, rff_x2_residual
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
  kernel.setup(X_train, n_top_comp=1024, n_fp_feat_budget=1024, residual_bit=0, bits_upperbound=32)
  kernel_mat_fp_pca_rff = kernel.get_kernel_matrix(X_train, X_train)

  np.testing.assert_array_almost_equal(kernel_mat_fp_rff.cpu().numpy(), kernel_mat_fp_pca_rff.cpu().numpy(), decimal=6)
  print("pca rff top k fp test passed!")

def pca_rff_top_k_lp_test():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]

  # get a full precision rff kernel
  kernel = GaussianKernel(sigma=sigma)
  kernel = RFF(32, n_input_feat, kernel, rand_seed=1)
  kernel_mat_fp_rff = kernel.get_kernel_matrix(X_train, X_train)

  # get a full precision PCA RFF
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFFTopK(32, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_top_comp=32, n_fp_feat_budget=32, residual_bit=0, bits_upperbound=32)
  quantizer = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=1, percentile=0.0)
  kernel_mat_fp_pca_rff = kernel.get_kernel_matrix(X_train, X_train, quantizer, quantizer)

  np.testing.assert_array_almost_equal(kernel_mat_fp_rff.cpu().numpy(), kernel_mat_fp_pca_rff.cpu().numpy(), decimal=6)
  print("pca rff top k fp test passed!")


def pca_rff_top_k_test():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]

  # get a full precision rff kernel
  kernel = GaussianKernel(sigma=sigma)
  kernel = RFF(32, n_input_feat, kernel, rand_seed=1)
  kernel_mat_fp_rff = kernel.get_kernel_matrix(X_train, X_train)

  # get a full precision PCA RFF
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFFTopK(32, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_top_comp=32, n_fp_feat_budget=32, residual_bit=0, bits_upperbound=32)
  quantizer = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=1, percentile=0.0)
  kernel_mat_fp_pca_rff = kernel.get_kernel_matrix(X_train, X_train, quantizer, quantizer)

  np.testing.assert_array_almost_equal(kernel_mat_fp_rff.cpu().numpy(), kernel_mat_fp_pca_rff.cpu().numpy(), decimal=6)
  print("pca rff top k fp test passed!")

def pca_rff_top_k_test_top_k():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]
  n_top_comp = 256

  quantizer1 = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=1, percentile=0.1)
  quantizer2 = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=2, percentile=0.1)

  # get a lp precision PCA RFF topK
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFFTopK(256, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_top_comp=n_top_comp, n_fp_feat_budget=64, residual_bit=0, bits_upperbound=32)
  kernel_mat_top_k = kernel.get_kernel_matrix(X_train, X_train, quantizer1, quantizer2)
  top_comp1_top_k = kernel.rff_x1_top_k.clone()
  top_comp2_top_k = kernel.rff_x2_top_k.clone()

  # get a lp precision PCA RFF topK
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFF(256, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_fp_feat_budget=64, bits_upperbound=32)
  kernel_mat_top_k = kernel.get_kernel_matrix(X_train, X_train, quantizer1, quantizer2)
  top_comp1_pca = kernel.comp_coor1.clone()
  top_comp2_pca = kernel.comp_coor2.clone()

  np.testing.assert_array_almost_equal(top_comp1_top_k.cpu().numpy(), 
    top_comp1_pca.cpu().numpy() )
  np.testing.assert_array_almost_equal(top_comp2_top_k.cpu().numpy(), 
    top_comp2_pca.cpu().numpy() )
  print("top K component test1 passed!")

def pca_rff_top_k_test_top_k2():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]
  n_top_comp = 48

  quantizer1 = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=1, percentile=0.1)
  quantizer2 = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=2, percentile=0.1)

  # get a lp precision PCA RFF topK
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFFTopK(256, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_top_comp=n_top_comp, n_fp_feat_budget=64, residual_bit=4, bits_upperbound=32)
  kernel_mat_top_k = kernel.get_kernel_matrix(X_train, X_train, quantizer1, quantizer2)
  top_k_assignment = kernel.bit_assignment_top_k
  top_comp1_top_k = kernel.rff_x1_top_k.clone()
  top_comp2_top_k = kernel.rff_x2_top_k.clone()

  # get a lp precision PCA RFF topK
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFF(256, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_fp_feat_budget=64, bits_upperbound=32)
  kernel.bit_assignment = top_k_assignment
  print("outside bit assignment", kernel.bit_assignment)
  kernel_mat_top_k = kernel.get_kernel_matrix(X_train, X_train, quantizer1, quantizer2)
  top_comp1_pca = kernel.comp_coor1.clone()
  top_comp2_pca = kernel.comp_coor2.clone()

  np.testing.assert_array_almost_equal(top_comp1_top_k[:, :n_top_comp].cpu().numpy(), 
    top_comp1_pca[:, :n_top_comp].cpu().numpy() )
  np.testing.assert_array_almost_equal(top_comp2_top_k[:, :n_top_comp].cpu().numpy(), 
    top_comp2_pca[:, :n_top_comp].cpu().numpy() )
  print("top K component test passed!")


def pca_rff_top_k_test_residual():
  data_path = "../../data/census/"
  sigma = 30.0
  X_train, X_test, Y_train, Y_test = load_census_data_part(data_path)
  n_input_feat = X_train.shape[1]
  n_top_comp = 48
  residual_bit = 8

  quantizer1 = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=1, percentile=0.1)
  quantizer2 = lambda nbit, min_val, max_val, rand_seed: \
          QuantizerAutoScale(nbit, min_val, max_val, 
          rand_seed=2, percentile=0.1)

  # get a lp precision PCA RFF topK
  kernel = GaussianKernel(sigma=sigma)
  kernel = PCA_RFFTopK(256, n_input_feat, kernel, rand_seed=1)
  kernel.setup(X_train, n_top_comp=n_top_comp, n_fp_feat_budget=128, residual_bit=residual_bit, bits_upperbound=32)
  kernel_mat_top_k = kernel.get_kernel_matrix(X_train, X_train, quantizer1=quantizer1, quantizer2=quantizer2)
  residual_raw1 = kernel.rff_x1_residual_raw.clone()
  residual_raw2 = kernel.rff_x2_residual_raw.clone()
  for i in range(residual_raw1.size(1) ):
    quant1 = quantizer1(residual_bit, 0, 0, 1)
    quant2 = quantizer2(residual_bit, 0, 0, 2)
    # if i == 1:
    #   print("residual before ", torch.min(residual_raw1[:, i] ), torch.max(residual_raw1[:, i] ) )
    residual_raw1[:, i] = quant1.quantize(residual_raw1[:, i])
    residual_raw2[:, i] = quant2.quantize(residual_raw2[:, i])
    # if i == 1:
    #   print("residual after ", torch.min(residual_raw1[:, i] ), torch.max(residual_raw1[:, i] ) )
  residual1 = kernel.rff_x1_residual.clone()
  residual2 = kernel.rff_x2_residual.clone()
  np.testing.assert_array_almost_equal(residual1.cpu().numpy(), residual_raw1.cpu().numpy() )
  np.testing.assert_array_almost_equal(residual2.cpu().numpy(), residual_raw2.cpu().numpy() )
  print("top K pca RFF residual quantization test passed!")


if __name__ == "__main__":
  pca_rff_top_k_test_residual()
  pca_rff_top_k_fp_test()
  pca_rff_top_k_lp_test()
  pca_rff_top_k_test_top_k()
  pca_rff_top_k_test_top_k2()

