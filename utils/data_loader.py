import numpy as np

# import scipy.io
# feat = scipy.io.loadmat("../../data/census/X_ho.npy")

# print feat
# print type(feat)


X_ho = np.load("../../data/census/X_ho.npy")
X_tr = np.load("../../data/census/X_tr.npy")
# # Y_ho = np.load("../../data/census/Y_ho.npy")
# # Y_tr = np.load("../../data/census/Y_tr.npy")

# print X_tr.item()
print np.mean(X_ho.item()['X_ho'], axis=0).shape#, np.mean(X_ho.item()['X_ho'], axis=0).shape
print np.mean(X_ho.item()['X_ho'], axis=1).shape#, np.mean(X_ho.item()['X_ho'], axis=1).shape
print np.mean(X_tr.item()['X_tr'], axis=0).shape#, np.mean(X_tr.item()['X_tr'], axis=0).shape
print np.mean(X_tr.item()['X_tr'], axis=1).shape#, np.mean(X_tr.item()['X_tr'], axis=1).shape



# res = np.sum(X_ho)
# print res
# print np.sum(X_ho["X_ho"]) #, np.sum(X_tr["X_tr"]), np.sum(Y_ho["Y_ho"] ), np.sum(Y_tr["Y_tr"] )