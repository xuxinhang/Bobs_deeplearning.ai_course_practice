# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

### Main Model | Start Here

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    params = initialize_parameters(layers_dims)

    ### Loop

    for i in range(0, num_iterations):

        if keep_prob >= 1:
            a3, cache = forward_propagation_my(X, params)
        else:
            a3, cache = forward_propagation_dropout(X, params, keep_prob)

        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_L2R(a3, Y, lambd, params)

        if lambd == 0 and keep_prob >= 1:
            grads = backward_propagation_my(X, Y, cache)
        elif keep_prob < 1:
            grads = backward_propagation_dropout(X, Y, cache, keep_prob)
        elif lambd != 0:
            grads = backward_propagation_L2R(X, Y, cache, lambd)
        else:
            grads = backward_propagation_my(X, Y, cache)

        params = update_parameters(params, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("iteration: %d \t Cost Value: %f" % (i, cost))

    return params

### def other function 

# L2正则化的损失函数
def compute_cost_L2R (A, Y, lambd, params):
    m = Y.shape[1]
    prev_cost = compute_cost(A, Y)
    added_item = (np.sum(np.square(params['W1'])) + np.sum(np.square(params['W1'])) + np.sum(np.square(params['W1']))) * lambd / m / 2
    return prev_cost + added_item

# 带有L2正则化的向后传播
def backward_propagation_L2R (X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T) +  lambd / m * W3
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(Z2 > 0))
    dW2 = 1/m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(Z1 > 0))
    dW1 = 1/m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
            "dZ1": dZ1, "dW1": dW1, "db1": db1}


# 我自己实现的初级 backward_propagation || 按照给出来的公式进行编写
def backward_propagation_my (X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(Z2 > 0))
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(Z1 > 0))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
            "dZ1": dZ1, "dW1": dW1, "db1": db1}


# 我自己的向前传播
def forward_propagation_my(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    Z1 = np.dot(W1,  X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    return A3, (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)



# Dropout 向前传播
def forward_propagation_dropout(X, params, keep_prob):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    np.random.seed(1)

    Z1 = np.dot(W1,  X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = A1 * D1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) < keep_prob
    A2 = A2 * D2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    return A3, (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

# Dropout 向后传播
def backward_propagation_dropout (X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2 / keep_prob
    dZ2 = np.multiply(dA2, np.int64(Z2 > 0))
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2) * D1 / keep_prob
    dA1 = dA1 * D1 / keep_prob
    dZ1 = np.multiply(dA1, np.int64(Z1 > 0))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
            "dZ1": dZ1, "dW1": dW1, "db1": db1}



### Run compute // 三种正则化的情况

train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Simple
parameters = model(train_X, train_Y)

# L2 正则化
# parameters = model(train_X, train_Y, lambd = 0.7)

# Dropout 正则化
# parameters = model(train_X, train_Y, keep_prob = 0.86)

print("Train dataset:\t")
predict(train_X, train_Y, parameters)
print("Test dataset:\t")
predict(test_X, test_Y, parameters)

# plot
plt.title("Model's Result")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

