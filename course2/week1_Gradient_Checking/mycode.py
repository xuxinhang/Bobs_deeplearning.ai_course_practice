###### 单层的梯度检测  ###### 

import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

forward_propagation  = lambda x, theta: x * theta
backward_propagation = lambda x, theta: x

def gradient_check (x, theta, epsilon = 1e-7):
    J = forward_propagation(x, theta)
    J_plus = forward_propagation(x, theta + epsilon)
    J_minus = forward_propagation(x, theta - epsilon)

    gradient_approx = (J_plus - J_minus) / (2 * epsilon)

    gradient_real = backward_propagation(x, theta)

    f = lambda x: np.sqrt(np.sum(np.square(x)))
    diff = f(gradient_approx - gradient_real) / (f(gradient_approx) + f(gradient_real))
    #### 注意欧几里得范数的定义，不要和前面的搞混了 ####
    # diff = np.linalg.norm(gradient_approx - gradient_real) / (np.linalg.norm(gradient_approx) + np.linalg.norm(gradient_real))
    
    if diff < 1e-7:
        return True, diff
    else:
        return False, diff



### RUN HERE

print("difference = {0}".format(gradient_check(2, 4)))







