import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector


def forward_propagation(X, Y, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']
    m = X.shape[1]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cost = 1./m * np.sum(np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y))
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backwrad_propagation(X, Y, cache):
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    m = X.shape[1]

    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)


    # TODO: 了解 np.int64
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return {"dZ3": dZ3, "dW3": dW3, "db3": db3,
            "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
            "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}



def check_grad_n (X, Y, params, epsilon = 1e-7):

    theta_raw, _ = dictionary_to_vector(params)
    param_num = theta_raw.shape[0]

    J, cache = forward_propagation(X, Y, params)

    # 手动计算梯度的估计值（参数数目：param_num）
    grad_approx = np.zeros((param_num, 1))
    for i in range(param_num):
        theta_minus = np.copy(theta_raw)
        theta_minus[i] -= epsilon
        theta_plus  = np.copy(theta_raw)
        theta_plus[i] += epsilon

        J_plus,  _ = forward_propagation(X, Y, vector_to_dictionary(theta_plus))
        J_minus, _ = forward_propagation(X, Y, vector_to_dictionary(theta_minus))

        grad_approx[i] = (J_plus - J_minus) / (2 * epsilon)

    grad = gradients_to_vector(backwrad_propagation(X, Y, cache))

    diff = np.linalg.norm(grad_approx - grad) / (np.linalg.norm(grad) + np.linalg.norm(grad))

    print((diff < 1e-7, diff))
    return diff



### RUN HERE

X, Y, parameters = gradient_check_n_test_case()
difference = check_grad_n(X, Y, parameters)

''' ## 最后的运行结果是和教程一致的 ## '''

