import numpy as np
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def sigma(x):
	return 1/(1+np.exp(-x))
def tanh_d(x):
	return 1-(np.tanh(x))**2

# 导入数据：花瓣数据/其他数据
planar = load_planar_dataset()
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
X, Y = planar
# 如果你不用planar数据，你需要添加下面这一行
# X, Y = X.T, Y.reshape(1, Y.shape[0])

# 初始化参数矩阵
W1 = np.random.randn(4,2) * 0.01
b1 = np.zeros((4,1))
W2 = np.random.randn(1,4) * 0.01
b2 = np.zeros((1,1))

# 需要的参数
m = Y.shape[1]
alaph = 1
A0 = X

# 开始迭代
for i in range(10000):
	# 正向传播
	Z1 = np.dot(W1, A0) + b1
	A1 = np.tanh(Z1)
	A2 = sigma(np.dot(W2, A1) + b2)
	# 反向传播
	J = -1/m * np.sum( np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2)) )
	dZ2 = A2 - Y
	dW2 = 1/m * np.dot(dZ2, A1.T)
	db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
	dW1 = 1/m * np.dot(dZ1, A0.T)
	db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
	# 梯度下降
	W2 = W2 - alaph * dW2
	W1 = W1 - alaph * dW1
	b2 = b2 - alaph * db2
	b1 = b1 - alaph * db1
	# 输出
	if i%1000 == 0:
		print("Loss:", J)
		print('W1', W1, '\nW2\n', W2)
		print('dW1', dW1, '\ndW2\n', dW2)
		print('b1', b1, 'b2', b2)


#  #  #  #  #  #
# 检验学习成果 #
#  #  #  #  #  #
pred_A1 = np.tanh(np.dot(W1, X) + b1)
pred_A2 = sigma(np.dot(W2, pred_A1) + b2)
pred_val = np.around(pred_A2)
fail_rate = np.sum(np.absolute(pred_val-Y)) / m
print('For training set, the fail rate is ', fail_rate*100)


# 预测函数
def predictor(inp_X):
	pred_A1 = np.tanh(np.dot(W1, inp_X) + b1)
	pred_A2 = sigma(np.dot(W2, pred_A1) + b2)
	pred_val = np.around(pred_A2)
	predictions = np.array( [1 if x >0.5 else 0 for x in A2.reshape(-1,1)] ).reshape(A2.shape) 
	return pred_val
# 预测某个点
predictor([[4],[2]])
# 绘制图形 in iPython-Notebook
%matplotlib inline
plot_decision_boundary(lambda x: predictor(x.T),X,Y)












