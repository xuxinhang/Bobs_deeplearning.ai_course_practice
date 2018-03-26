import time
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# arrange dataset
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


def init_params (layer_dims):
	np.random.seed(3)
	params = {}
	L = len(layer_dims)
	for l in range(1,L):
		params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		params['b' + str(l)] = np.zeros((layer_dims[l], 1))
	return params

def linear_forward (A, W, b):
	Z = np.dot(W,A) + b
	midval = (A,W,b)
	return Z, midval

def acti_forward (A, W, b, which_acti):
	Z, linear_midval = linear_forward(A,W,b)
	if which_acti == 'sigmoid':
		A, acti_midval = sigmoid(Z)
	elif which_acti == 'relu':
		A, acti_midval = sigmoid(Z)

	return A, (linear_midval, acti_midval)

def layer_forward (X, params):
	midvals = []
	A = X
	L = len(params) // 2
	for l in range(1, L+1):
		W = params['W' + str(l)]
		b = params['b' + str(l)]
		A, midval = acti_forward(A, W, b, which_acti = 'sigmoid' if l==L else 'relu')
		midvals.append(midval)
	
	return A, midvals
	
def compute_cost_func (Y_hat, Y):
	m = Y.shape[1]
	J = -1/m * np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))
	return J

def linear_backward (dZ, midval):
	A_prev, W, b = midval
	m = A_prev.shape[1]
	dA_prev = np.dot(W.T, dZ)
	db = 1/m * np.sum(dZ, axis=1, keepdims=True)
	dW = 1/m * np.dot(dZ, A_prev.T)
	return dA_prev, dW, db

def acti_backward (dA, midval, which_acti):
	linear_midval, activation_midval = midval
	if which_acti == "relu":
		dZ = sigmoid_backward(dA, activation_midval)
		dA_prev, dW, db = linear_backward(dZ, linear_midval)     
	elif which_acti == "sigmoid":
		dZ = sigmoid_backward(dA, activation_midval)
		dA_prev, dW, db = linear_backward(dZ, linear_midval)
	return dA_prev, dW, db
	
def layer_backward (A, Y, midvals):
	grads = {}
	L = len(midvals)
	m = A.shape[1]
	dA = -(np.divide(Y, A) - np.divide(1-Y, 1-A))
	grads['dA' + str(L+1)] = dA
	for l in range(L,0,-1):
		grads['dA' + str(l)], grads['dW' + str(l)], grads['db' + str(l)] = acti_backward(grads['dA' + str(l+1)], midvals[l-1], 'sigmoid' if l==L else 'relu')
	
	return grads

def update_params (params, grads, learning_rate):
	L = len(params) // 2
	for l in range(L):
		params['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
		params['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]

	return params


def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000):
	np.random.seed(1)
	costs = []
	params = init_params(layers_dims)
	for i in range(0, num_iterations):
		AL, midvals = layer_forward(X, params)
		cost = compute_cost_func(AL, Y)
		grads = layer_backward(AL, Y, midvals)
		params= update_params(params, grads, learning_rate)
		if i % 100 == 0:
			print ("Cost Func :: %i %f" % (i, cost))
		
	return params

#START
 
parameters = model(train_x, train_y, [train_x.shape[0], 20, 7, 5, 1])
print('Params::' + str(parameters))

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


