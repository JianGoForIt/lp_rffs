import numpy as np

def load_census_data(path="../../data/census/"):
  X_test = np.load(path + "X_ho.npy")
  X_train = np.load(path + "X_tr.npy")
  Y_test = np.load(path + "Y_ho.npy")
  Y_train = np.load(path + "Y_tr.npy")
  X_test = X_test.item()['X_ho']
  X_train = X_train.item()['X_tr']
  Y_test = Y_test.item()['Y_ho']
  Y_train = Y_train.item()['Y_tr']
  # # DEBUG
  # X_train, Y_train, X_test, Y_test = \
  #   X_train[1:20, :], Y_train[1:20], X_test[1:10, :], Y_test[1:10]
  return X_train, X_test, Y_train, Y_test

def load_census_data_part(path):
  X_test = np.load(path + "X_ho.npy")
  X_train = np.load(path + "X_tr.npy")
  Y_test = np.load(path + "Y_ho.npy")
  Y_train = np.load(path + "Y_tr.npy")
  X_test = X_test.item()['X_ho']
  X_train = X_train.item()['X_tr']
  Y_test = Y_test.item()['Y_ho']
  Y_train = Y_train.item()['Y_tr']
  s = np.arange(X_train.shape[0] )
  np.random.seed(0)
  np.random.shuffle(s)
  X_train = X_train[s, :]
  Y_train = Y_train[s]
  X_train, Y_train, X_test, Y_test = \
    X_train[:(s.size * 1 / 5), :], Y_train[:(s.size * 1 / 5)], X_test[:(s.size * 1 / 5), :], Y_test[:(s.size * 2 / 3)]
  return X_train, X_test, Y_train, Y_test



# # import scipy.io
# # feat = scipy.io.loadmat("../../data/census/X_ho.npy")

# # print feat
# # print type(feat)


# X_ho = np.load("../../data/census/X_ho.npy")
# X_tr = np.load("../../data/census/X_tr.npy")
# # # Y_ho = np.load("../../data/census/Y_ho.npy")
# # # Y_tr = np.load("../../data/census/Y_tr.npy")

# # print X_tr.item()
# print np.mean(X_ho.item()['X_ho'], axis=0).shape#, np.mean(X_ho.item()['X_ho'], axis=0).shape
# print np.mean(X_ho.item()['X_ho'], axis=1).shape#, np.mean(X_ho.item()['X_ho'], axis=1).shape
# print np.mean(X_tr.item()['X_tr'], axis=0).shape#, np.mean(X_tr.item()['X_tr'], axis=0).shape
# print np.mean(X_tr.item()['X_tr'], axis=1).shape#, np.mean(X_tr.item()['X_tr'], axis=1).shape



# # res = np.sum(X_ho)
# # print res
# # print np.sum(X_ho["X_ho"]) #, np.sum(X_tr["X_tr"]), np.sum(Y_ho["Y_ho"] ), np.sum(Y_tr["Y_tr"] )