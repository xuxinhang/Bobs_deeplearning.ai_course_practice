import time
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *


def tanh (z):
	return np.tanh(z)
def tanh_d(x):
	return 1-np.power(np.tanh(x), 2)
def relu (z):
	return np.maximum(0, z)
def sigma (z):
	return 1/(1+np.exp(-z))
def sigma_d (z):
	temp = sigma(z)
	return temp*(1-temp)
def relu_d (dA):
	dZ = np.array(dA, copy=True)
	tf = np.greater_equal(0, dA)
	dZ[tf] = 0
	return dZ
def acti_func (z, f='tanh'):
	res = 0
	if f == 'relu':
		res = relu(z)
	elif f == 'sigma':
		res = sigma(z)
	elif f == 'tanh':
		res = tanh(z)
	return res
def acti_fund (z, f='tanh'):
	res = 0
	if f == 'relu':
		res = relu_d(z)
	elif f == 'sigma':
		res = sigma_d(z)
	elif f == 'tanh':
		res = tanh_d(z)
	return res


# 导入数据
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

X = train_x
Y = train_y


m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T /255.
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T /255.




# 初始化参数
m = X.shape[1]
learning_rate = 0.1
repeat_number = 1000
layer_dims = [X.shape[0], 20, 7, 1]
layer_func = ['', 'tanh', 'tanh', 'sigma', 'sigma']
layer_number = len(layer_dims)
layer_count = layer_number - 1
# 初始化矩阵参数
A = list(range(layer_number))
W = list(range(layer_number))
b = list(range(layer_number))
Z = list(range(layer_number))
dZ = list(range(layer_number))
db = list(range(layer_number))
dW = list(range(layer_number))
dA = list(range(layer_number))
A[0] = X
for i in range(1, layer_number):
	b[i] = np.zeros((layer_dims[i] ,1))
	W[i] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
print("layer_dims::" + str(layer_dims))

# 开始迭代
print('count:' + str(layer_count))
print('m:' + str(m))
for ii in range(repeat_number):
	# 正向传播
	for i in range(1, layer_number):
		Z[i] = np.dot(W[i], A[i-1]) + b[i]
		A[i] = acti_func(Z[i], layer_func[i])
	AL = A[layer_count]
	# 反向传播
	J = -1/m * np.sum( Y*np.log(AL)+(1-Y)*(np.log(1-AL)))
	Y = Y.reshape(AL.shape)
	dA[layer_count] = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
	for i in range(layer_count, 0, -1):
		dZ[i] = dA[i] * acti_fund(Z[i], layer_func[i])
		dA[i-1] = np.dot(W[i].T, dZ[i])
		db[i] = 1/m * np.sum(dZ[i], axis=1, keepdims=True)
		dW[i] = 1/m * np.dot(dZ[i], A[i-1].T)
	# 更新参数
	for i in range(1, layer_number):
		W[i] = W[i] - learning_rate * dW[i]
		b[i] = b[i] - learning_rate * db[i]
	# 辅助输出
	if ii%100 == 0:
		print('Loss func [J] ::' + str(J))


# 检验
for i in range(1, layer_number):
	Z[i] = np.dot(W[i], A[i-1]) + b[i]
	A[i] = acti_func(Z[i], layer_func[i])
Y_hat = np.zeros((1, X.shape[1]))
AL = A[layer_count]
for i in range(AL.shape[1]):
  Y_hat[0][i] = AL[0][i] > 0.5
fail_counter = np.sum(np.absolute(Y_hat - Y))
print("Fail rate : ", fail_counter / Y.shape[1])	


print('W', W)
print('b', b)



