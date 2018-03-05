import numpy as np
import torch

class Quantizer(object):
  def __init__(self, nbit, min_val, max_val, scale=None):
    self.nbit = nbit
    self.min_val = min_val
    self.max_val = max_val
    if scale == None:
      self.scale = (max_val - min_val) / float(2**self.nbit - 1)

  def quantize_random(self, value):
    value = torch.clamp(value, self.min_val, self.max_val)
    floor_val = self.min_val + torch.floor( (value - self.min_val) / self.scale) * self.scale
    ceil_val = self.min_val + torch.ceil( (value - self.min_val) / self.scale) * self.scale
    floor_prob = ceil_val - value
    ceil_prob = value - floor_val
    # sanity check
    np.testing.assert_array_almost_equal(floor_prob.cpu().numpy(), 
      1 - ceil_prob.cpu().numpy() )
    sample = torch.FloatTensor(np.random.uniform(size=list(value.size() ) ) )
    quant_val = floor_val * (sample < floor_prob).float() \
      + ceil_val * (sample >= floor_prob).float()
    return quant_val


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


if __name__ == "__main__":
  test_random_quantizer()



