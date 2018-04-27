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
		if isinstance(X, np.ndarray):
			X = torch.FloatTensor(X)
		perm = torch.randperm(X.size(0) )
		# using the standard way to select n_feat landmark points
		if n_landmark is None:
			n_landmark = self.n_feat
		self.landmark = X[perm[:n_landmark], :]
		self.n_landmark = n_landmark
		self.K_landmark = \
			self.kernel.get_kernel_matrix(self.landmark.double(), self.landmark.double() )
		# linalg.eigh can give negative value on cencus regression dataset
		# So we use svd here and we have not seen numerical issue yet.
		S, U = np.linalg.eigh(self.K_landmark.cpu().numpy().astype(np.float64), UPLO='U')
		S = S[::-1].copy()
		U = U[:, ::-1].copy()
		if np.min(S[:self.n_landmark] ) <= 0:
			print("numpy eigh gives negative values, switch to use SVD")
			U, S, _ = np.linalg.svd(self.K_landmark.cpu().numpy().astype(np.float64) )

		self.U_d = torch.FloatTensor(U[:, :n_landmark] )
		self.S_d = torch.FloatTensor(S[:n_landmark] )
		self.A_d = torch.mm(self.U_d, torch.diag(1.0/torch.sqrt(self.S_d) ) )
		# print "inside ", torch.sum(self.U_d), torch.sum(self.S_d), torch.sum(self.A_d)
		# print self.S_d, torch.sqrt(self.S_d)
		# # print torch.diag(1.0/torch.sqrt(self.S_d) )



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


# test full dimension almost match exact kernel results
def test_nystrom_full():
	# test if keep all the dimensions is the nystrom kernel matrix equals to the exact kernel
	n_sample = 15
	n_feat = n_sample
	input_val1  = np.random.normal(size=[n_sample, n_feat] )
	input_val2  = np.random.normal(size=[n_sample - 1, n_feat] )
	# get exact gaussian kernel
	kernel = GaussianKernel(sigma=np.random.normal() )
	kernel_mat = kernel.get_kernel_matrix(input_val1, input_val2)

	approx = Nystrom(n_feat, kernel=kernel)
	approx.setup(input_val1)
	approx_kernel_mat = approx.get_kernel_matrix(torch.Tensor(input_val1), torch.Tensor(input_val2) )

	np.testing.assert_array_almost_equal(kernel_mat.cpu().numpy(), approx_kernel_mat.cpu().numpy() )
	print("nystrom full dimension test passed!")


if __name__ == "__main__":
	test_nystrom_full()