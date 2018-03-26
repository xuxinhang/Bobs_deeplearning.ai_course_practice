import numpy as np
import h5py
from testCases_v2 import *
from lr_utils import load_dataset
# from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
np.random.seed(1)

# 导入数据集 Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_number = train_set_x_orig.shape[0]
train_set_x_flat = train_set_x_orig.reshape(train_set_number, -1).T
X = train_set_x_flat/255
Y = train_set_y


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

def acti_func (z, f='relu'):
	res = 0
	if f == 'relu':
		res = relu(z)
	elif f == 'sigma':
		res = sigma(z)
	elif f == 'tanh':
		pass
	return res
	
def acti_fund (z, f='relu'):
	res = 0
	if f == 'relu':
		res = relu_d(z)
	elif f == 'sigma':
		res = sigma_d(z)
	elif f == 'tanh':
		pass
	return res


# 初始化超参数
layer_number = 0
learning_rate = 0.01
repeat_number = 5000
layer_dims = [X.shape[0], 1, 3,1]
layer_func = ['', 'relu', 'relu', 'sigma']
m = X.shape[1]

# 初始化参数
layer_number = len(layer_dims)
layer_count = layer_number - 1
A = list(range(layer_number))
W = list(range(layer_number))
b = list(range(layer_number))
Z = list(range(layer_number))
dZ = list(range(layer_number))
db = list(range(layer_number))
dW = list(range(layer_number))
dA = list(range(layer_number))
for i in range(1, layer_number):
	b[i] = np.zeros((layer_dims[i] ,1))
	W[i] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
A[0] = X


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
	dA[layer_count] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	for i in range(layer_count, 0, -1):
		dZ[i] = dA[i] * acti_fund(Z[i], layer_func[i])
		dA[i-1] = np.dot(W[i].T, dZ[i])
		db[i] = 1/m * np.sum(dZ[i], axis=1, keepdims=True)
		dW[i] = 1/m * np.dot(dZ[i], A[i-1].T)
	# 更新参数
	for i in range(1, layer_number):
		W[i] -= learning_rate * dW[i]
		b[i] -= learning_rate * db[i]
	# 辅助
	if (ii%100 == 0):
		print('Loss func::' + str(J))
		# print('b::' + str(b))


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






