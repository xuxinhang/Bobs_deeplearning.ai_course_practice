import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



# 最简单的梯度下降法
def update_params_batch (params, grads, learning_rate):
    L = len(params) // 2  # number of layer

    for l in range(1, L+1): 
        params['W%d' % l] -= learning_rate * grads['dW%d' % l]
        params['b%d' % l] -= learning_rate * grads['db%d' % l]

    return params

# 构建 MiniBatch 数据
def random_mini_batches(X, Y, each_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    batches = []
    batch_number = m // each_size

    arrange = np.random.permutation(m)
    sX = X[:, arrange]
    sY = Y[:, arrange]

    for k in range(0, batch_number):
        start_index = k*each_size
        end_index = start_index + each_size
        batches += [(sX[:, start_index:end_index], sY[:, start_index:end_index])]

    if m % each_size != 0:
        k = batch_number
        start_index = k * each_size
        end_index = m
        batches.append((sX[:, start_index:end_index], sY[:, start_index:end_index]))

    return batches


def test_random_mini_batches(): 
    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

    print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

# test_random_mini_batches() # PASS!

def initialize_velocity(params):
    L = len(params) // 2   # how many layers 
    v = {}

    for l in range(1, L+1):
        v['dW%d' % l] = np.zeros(params['W%d' % l].shape)
        v['db%d' % l] = np.zeros(params['b%d' % l].shape)

    return v

def test_initialize_velocity():
    # print(initialize_velocity_test_case())
    v = initialize_velocity(initialize_velocity_test_case())
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))

# test_initialize_velocity() #PASS


# Momentum 方法
def update_params_momt(params, grads, v, beta, learning_rate):
    L = len(params) // 2  # number of layer
    # v = initialize_velocity(params)

    for l in range(1, L+1):
        v['dW%d' % l] = beta * v['dW%d' % l] + (1-beta) * grads['dW%d' % l]
        params['W%d' % l] -= learning_rate * v['dW%d' % l]
        v['db%d' % l] = beta * v['db%d' % l] + (1-beta) * grads['db%d' % l]
        params['b%d' % l] -= learning_rate * v['db%d' % l]

    return params, v

def test_update_params_momt():
    parameters, grads, v = update_parameters_with_momentum_test_case()
    parameters, v = update_params_momt(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))

# test_update_params_momt() # PASS


# Adeam 方法
def initialize_adam(params):
    L = len(params) // 2   # how many layers 
    v = {}
    s = {}

    for l in range(1, L+1):
        v['dW%d' % l] = np.zeros(params['W%d' % l].shape)
        v['db%d' % l] = np.zeros(params['b%d' % l].shape)
        s['dW%d' % l] = np.zeros(params['W%d' % l].shape)
        s['db%d' % l] = np.zeros(params['b%d' % l].shape)

    return v, s

def update_params_adam (params, grads, v, s, t, learning_rate = 0.01,
                        beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    L = len(params) // 2  # number of layer
    # v, s = initialize_adam(params)
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):
        v['dW%d' % l] = (beta1 * v['dW%d' % l] + (1-beta1) * grads['dW%d' % l])
        v['db%d' % l] = (beta1 * v['db%d' % l] + (1-beta1) * grads['db%d' % l])
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)

        s['dW%d' % l] = (beta2 * s['dW%d' % l] + (1-beta2) * grads['dW%d' % l]**2)
        s['db%d' % l] = (beta2 * s['db%d' % l] + (1-beta2) * grads['db%d' % l]**2)
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        params['W%d' % l] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        params['b%d' % l] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return params, v, s

def test_update_params_adam():
    parameters, grads, v, s = update_parameters_with_adam_test_case()
    parameters, v, s  = update_params_adam(parameters, grads, v, s, t = 2)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))

# test_update_params_adam() # Some difference



### Model Basic

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):

    L = len(layers_dims)
    seed = 10
    params = initialize_parameters(layers_dims)

    if optimizer == 'momt':
        v = initialize_velocity(params)
    elif optimizer == 'adam':
        v, s = initialize_adam(params)
    else:
        pass

    for i in range(num_epochs):
        seed += 1
        batches = random_mini_batches(X, Y, mini_batch_size, seed)

        for batch in batches:
            (batch_X, batch_Y) = batch
            A, cache = forward_propagation(batch_X, params)
            J = compute_cost(A, batch_Y)
            grads = backward_propagation(batch_X, batch_Y, cache)

            if optimizer == 'momt':
                params, v = update_params_momt(params, grads, v, beta, learning_rate)
            elif optimizer == 'adam':
                params, v, s = update_params_adam(params, grads, v, s, i+1, learning_rate, beta1, beta2, epsilon)
            else:
                params = update_params_batch(params, grads, learning_rate)
    
        if print_cost and i % 1000 == 0:
            print("iter: %d\tcost: %f" % (i, J))

    return params



# Run model
train_X, train_Y = load_dataset()
layers_dims = [train_X.shape[0], 5, 2, 1]

# parameters = model(train_X, train_Y, layers_dims, 'batch')   #PASS
# parameters = model(train_X, train_Y, layers_dims, 'momt')   #PASS
parameters = model(train_X, train_Y, layers_dims, 'adam')   #PASS

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


'''
总体来说还是很成功的
'''















