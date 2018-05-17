"""
本周第二个任务

front_propagation 函数的结果总是不对，即使是按照 GitHub 别人的代码
我修改了forward_propagation函数中的一些参数。 令人惊讶的是，代码获得了更好的结果，并且准确性接近1 ！
"""


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
np.random.seed(1)

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], 'X')
    Y = tf.placeholder(tf.float32, [None, n_y], 'Y')
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', [4,4,3,8], tf.float32, tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', [2,2,8,16], tf.float32, tf.contrib.layers.xavier_initializer(seed = 0))

    return {'W1': W1, 'W2': W2}


def forward_propagation(X, params):
    # 第一个 max_pool 参数被修改了，结果惊喜地取得了比原文还要好的准确度！
    W1 = params['W1']
    W2 = params['W2']

    Z1 = tf.nn.conv2d(X, W1, [1,1,1,1], 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, [1,2,2,1], [1,2,2,1], 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, [1,1,1,1], 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, [1,4,4,1], [1,4,4,1], 'SAME')
    F2 = tf.contrib.layers.flatten(P2)
    FC = tf.contrib.layers.fully_connected(F2, activation_fn=None, num_outputs=6)

    return FC


def compute_cost(FC, Y):
    m, n_y = Y.shape
    single_J = tf.nn.softmax_cross_entropy_with_logits(logits = FC, labels = Y)
    return tf.reduce_mean(single_J)


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    m, n_H0, n_W0, n_C0 = X_train.shape
    m, n_y = Y_train.shape
    cost_record = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    params = initialize_parameters()
    FC = forward_propagation(X, params)
    J = compute_cost(FC, Y)
    opti = tf.train.AdamOptimizer(learning_rate = learning_rate)
    opti_op = opti.minimize(J)
    init_op = tf.global_variables_initializer()


    minibatch_num = m // minibatch_size 

    with tf.Session() as session:
        session.run(init_op)
        session.run(params)

        for i in range(num_epochs): # Interite main loop
            minibatch_cost = 0
            seed += 1
            np.random.seed(seed)
            arrange = np.random.permutation(m) # minibatch
            X_batch = X_train[arrange]
            Y_batch = Y_train[arrange]

            for b in range(0, m, minibatch_size):
                _, current_cost = session.run([opti_op, J], feed_dict = {
                    X: X_train[b:b+minibatch_size],
                    Y: Y_train[b:b+minibatch_size],
                })
                minibatch_cost += current_cost

            minibatch_cost /= minibatch_num
            if print_cost == True and i % 5 == 0:
                print ("Cost after epoch %i: %f" % (i, minibatch_cost))
            # if print_cost == True and epoch % 1 == 0:
            #     costs.append(minibatch_cost)

        # Calculate the correct predictions
        predict_op = tf.argmax(FC, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train[0:100], Y: Y_train[0:100]})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters


# Start to run this model.
_, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=90)






