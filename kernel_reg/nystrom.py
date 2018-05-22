import numpy as np
import scipy
import torch
from rff import GaussianKernel

EPS = 1e-15

class Nystrom(object):
	def __init__(self, n_feat, kernel=None, rand_seed=1):
		self.n_feat = n_feat
		self.kernel = kernel
		self.rand_seed = rand_seed

	def setup(self, X, n_landmark=None):
		'''
		X is in the shape of [n_sample, n_dimension]
		call setup() once before using Nystrom
		'''
		# if n feat > n sample then make n feat = n sample
		if self.n_feat > X.size(0):
			self.n_feat = X.size(0)
		np.random.seed(self.rand_seed)
		perm = np.random.permutation(np.arange(X.size(0) ) )
		# using the standard way to select n_feat landmark points
		if n_landmark is None:
			n_landmark = min(self.n_feat, X.size(0) )
			print("# landmarks ", n_landmark)
		self.landmark = X[perm[:n_landmark], :]
		self.n_landmark = n_landmark
		#self.landmark = X
		#self.n_landmark = X.size(0)
		self.K_landmark = \
			self.kernel.get_kernel_matrix(self.landmark, self.landmark)
		# # the torch mm function can make very sutle difference of upper and lower triangular in
		# # a symetric matrix
		# if not torch.equal(self.K_landmark, torch.transpose(self.K_landmark, 0, 1) ):
		# 	raise Exception("Kernel matrix is not symetric!")
		# linalg.eigh can give negative value on cencus regression dataset
		# So we use svd here and we have not seen numerical issue yet.
		#S, U = np.linalg.eigh(self.K_landmark.cpu().numpy().astype(np.float64), UPLO='U')
		#S = S[::-1].copy()
		#U = U[:, ::-1].copy()
		#if np.min(S[:self.n_landmark] ) <= 0:
		#	print("numpy eigh gives negative values, switch to use SVD")
		U, S, _ = np.linalg.svd(self.K_landmark.cpu().numpy() )
		#U = np.random.normal(size=(self.K_landmark.size(0), self.K_landmark.size(1)))
		#S = np.random.normal(size=(self.K_landmark.size(0)) )
		self.U_d = torch.DoubleTensor(U[:, :n_landmark] )
		self.S_d = torch.DoubleTensor(S[:n_landmark] )
		self.A_d = torch.mm(self.U_d, torch.diag(1.0/torch.sqrt(self.S_d) ) )

	def get_feat(self, X):
		kernel_matrix = self.kernel.get_kernel_matrix(X, self.landmark)
		feat = torch.mm(kernel_matrix, self.A_d)
		return feat

	def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None):
		feat_x1 = self.get_feat(X1)
		feat_x2 = self.get_feat(X2)
		return torch.mm(feat_x1, torch.transpose(feat_x2, 0, 1) )

	def torch(self, cuda):
		if cuda:
			self.A_d = self.A_d.cuda()
			self.landmark = self.landmark.cuda()

	def cpu(self):
		self.A_d = self.A_d.cpu()
		self.landmark = self.landmark.cpu()

# test full dimension almost match exact kernel results
def test_nystrom_full():
	# test if keep all the dimensions is the nystrom kernel matrix equals to the exact kernel
	n_sample = 15
	n_feat = n_sample
	input_val1  = torch.Tensor(np.random.normal(size=[n_sample, n_feat] ) ).double()
	input_val2  = torch.Tensor(np.random.normal(size=[n_sample - 1, n_feat] ) ).double()
	# get exact gaussian kernel
	kernel = GaussianKernel(sigma=np.random.normal() )
	kernel_mat = kernel.get_kernel_matrix(input_val1, input_val2)

	approx = Nystrom(n_feat, kernel=kernel)
	approx.setup(input_val1)
	approx_kernel_mat = approx.get_kernel_matrix(input_val1, input_val2)

	np.testing.assert_array_almost_equal(kernel_mat.cpu().numpy(), approx_kernel_mat.cpu().numpy() )
	print("nystrom full dimension test passed!")


if __name__ == "__main__":
	test_nystrom_full()
