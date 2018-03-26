import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


train_X, train_Y, test_X, test_Y = load_dataset()


def main (X, Y, alaph = 0.01, iteration_num = 15000, how_to_init_params = 0):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 14, 8, 1]
    if how_to_init_params == 0: 
        params = initialize_parameters_zeros(layers_dims)  # zeros
    elif how_to_init_params == 1:
        params = initialize_parameters_random(layers_dims)  # randn
    else:
        params = initialize_parameters_opt(layers_dims)  # new

    # iteration loop
    for i in range(0, iteration_num):
        a3, cache = forward_propagation(X, params)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        params = update_parameters(params, grads, alaph)
        # print the loss
        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            # costs.append
    return params


def initialize_parameters_zeros (dims):
    params = {}
    L = len(dims)
    for l in range(1, L):
        params["W%d" % l] = np.zeros((dims[l], dims[l-1]))
        params["b%d" % l] = np.zeros((dims[l], 1))
    return params

def initialize_parameters_random (dims):
    params = {}
    L = len(dims)
    for l in range(1, L):
        params["W%d" % l] = np.random.randn(dims[l], dims[l-1]) * 1 #10
        # 这里乘10是为了让效果更明显，讲道理是不应该乘10的
        params["b%d" % l] = np.zeros((dims[l], 1))
    return params

def initialize_parameters_opt (dims):
    params = {}
    L = len(dims)
    for l in range(1, L):
        params["W%d" % l] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2/dims[l-1])
        params["b%d" % l] = np.zeros((dims[l], 1))
    return params

for way in range(0,3):
    print("= = = = = = = = = = = = \n = = = = =" + ['zeros', 'randn', 'new'][way] + "= = = = = \n = = = = = = = = = = = =")
    result = main(train_X, train_Y, how_to_init_params = way, iteration_num = 25000)
    # check success rate
    print ("train dataset:\n" + str(predict(train_X, train_Y, result)))
    print ("test  dataset:\n" + str(predict(test_X, test_Y, result)))
    # draw
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(result, x.T), train_X, train_Y)
