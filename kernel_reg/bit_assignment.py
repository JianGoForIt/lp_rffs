import numpy as np

def get_bits(spectrum, s, upper_bound=None):
  '''
  spectrum is the singular values of RFF matrix, not of the kernel matrix
  '''
  assert type(upper_bound) == int
  nbits = np.maximum(np.log2(spectrum**2) + s, np.zeros(spectrum.size) )
  if upper_bound != None:
    nbits = np.minimum(nbits, upper_bound * np.ones(spectrum.size) )
  nbits = np.floor(nbits) 
  #nbits[nbits!=0] = 2**(np.floor(np.log2(nbits[nbits!=0] ) ) )
#     return 2**(np.floor(np.log2(nbits) ) ).astype(np.int)
#     return np.exp(np.log2(nbits) ).astype(np.int)
#     return np.floor(nbits)
  return nbits.astype(np.int)
    
def get_n_fp_feat(spectrum, s, upper_bound=None):
  nbits = get_bits(spectrum, s, upper_bound)
  n_fp_feat = nbits.astype(np.float) / 32.0
  return np.sum(n_fp_feat), nbits 

def binary_search_bits_assignment(spectrum, n_fp_feat_budget, left=-2000.0, right=2000.0, tolerance=1e-6, upper_bound=None):
  # upper_bound is the upper bound of number of bits
  n_fp_feat = 0
  best_n_fp_feat = 0
  best_sol = None
  while right - left >= tolerance:
    middle = (left + right) / 2.0
    n_fp_feat, nbits = get_n_fp_feat(spectrum, middle, upper_bound)
    if n_fp_feat == n_fp_feat_budget:
      print("found exact assignment plan", nbits)
      best_sol = nbits
      best_n_fp_feat = n_fp_feat
      break
    elif n_fp_feat < n_fp_feat_budget:
      left = middle
    else:
      right = middle
    if best_n_fp_feat < n_fp_feat and n_fp_feat < n_fp_feat_budget:
      best_sol = nbits
      best_n_fp_feat = n_fp_feat
#         print("testing middle = ", middle, " n_fp_feat ", n_fp_feat)
  print("best solution, memory budget ", best_n_fp_feat, " / ", n_fp_feat_budget)
  return best_sol

def test_binary_search_bit_assignment1():
  # enough memory budget can host all upperbound bits
  file_name = "./multi_seed_results/spectrum/s_8192_feat_fp_64b_svd_ind_full_dataset.npy"
  with open(file_name, "rb") as f:
      spectrum = np.load(f) 
  spectrum = spectrum[:int(file_name.split("/")[-1].split("_")[1] ) ]

  n_fp_feat_budget=8192
  search_left = -200.0
  search_right = 200.0
  tolerance=1e-6
  upper_bound = 32

  nbits = binary_search_bits_assignment(spectrum, n_fp_feat_budget, 
                                        search_left, search_right,
                                        tolerance, upper_bound)
  assert np.max(nbits) == upper_bound
  assert np.min(nbits) == upper_bound
  print("binary search bit assignment test 1 done")
  return 


def test_binary_search_bit_assignment2():
  # make sure we can have enough bits to host the assignment
  file_name = "./multi_seed_results/spectrum/s_8192_feat_fp_64b_svd_ind_full_dataset.npy"
  with open(file_name, "rb") as f:
    spectrum = np.load(f) 
  spectrum = spectrum[:int(file_name.split("/")[-1].split("_")[1] ) ]

  n_fp_feat_budget=1024
  search_left = -200.0
  search_right = 200.0
  tolerance=1e-6
  upper_bound = 32

  nbits = binary_search_bits_assignment(spectrum, n_fp_feat_budget, 
                                      search_left, search_right,
                                      tolerance, upper_bound)
  # print nbits, np.sum(nbits) / 32.0
  assert np.sum(nbits) / 32.0 <= n_fp_feat_budget
  print("binary search bit assignment test 2 done")
  return 


if __name__ == "__main__":
  test_binary_search_bit_assignment1()
  test_binary_search_bit_assignment2()

