# import numpy as np
# import scipy
# import matplotlib
#
# from pprint import pprint
# from scipy.io import loadmat
# from sklearn.datasets import fetch_mldata
#
#
# import time
# import matplotlib.pyplot as plt
# import numpy as np
#
# from sklearn.datasets import fetch_mldata
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import check_random_state
#
# print(__doc__)
#
# # Author: Arthur Mensch <arthur.mensch@m4x.org>
# # License: BSD 3 clause
#
# # Turn down for faster convergence
# t0 = time.time()
# train_samples = 6000
#
# mnist = fetch_mldata('MNIST original')
# X = mnist.data.astype('float64')
# y = mnist.target
# random_state = check_random_state(0)
# permutation = random_state.permutation(X.shape[0])
# X = X[permutation]
# y = y[permutation]
# X = X.reshape((X.shape[0], -1))
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=train_samples, test_size=1000)
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# print(X_test)
# print(X_train)
#
# mnist = fetch_mldata('MNIST original', data_home="/Users/naitikshah/PycharmProjects/KNN")
# print(mnist)
# print(mnist.data.shape)
# print(mnist.target.shape)
#
# print(np.unique(mnist.target))

import pandas as pd

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:6000], X[6000:7000]
y_train, y_test = y[:6000], y[6000:7000]


print(len(X_train))
print(len(X_test))

print(pd.DataFrame(X_train))



