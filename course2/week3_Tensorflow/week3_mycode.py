import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
np.random.seed(1)


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


def my_model (X_set, Y_set, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_set.shape                            # (n_x: input size, m : number of examples in the train set)
    n_y = Y_set.shape[0]
    
    costs = []

    # initialize Placeholder [VERIFIED]
    tf.set_random_seed(1)
    X = tf.placeholder(tf.float32, shape=[n_x, None], name = 'X')
    Y = tf.placeholder(tf.float32, [n_y, None], name = 'Y')

    # initialize params
    W1 = tf.get_variable("W1", [25,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())

    # compute forward propgation
    A1 = tf.nn.relu(tf.matmul(W1, X) + b1, name = "A1")
    A2 = tf.nn.relu(tf.matmul(W2, A1) + b2, name = "A2")
    Z3 = tf.matmul(W3, A2) + b3

    # Cost function 
    logits = tf.transpose(Z3) # WHY?
    labels = tf.transpose(Y)
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    # Optimizer Trainer
    opti = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(J)
    # AdamOptimizer

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(num_epochs):

        # print('i')
        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        seed = seed + 1
        minibatches = random_mini_batches(X_set, Y_set, minibatch_size, seed)

        for (minibatch_X, minibatch_Y) in minibatches:
            _, minibatch_cost = sess.run([opti, J], feed_dict={X: minibatch_X, Y: minibatch_Y})
            epoch_cost += minibatch_cost / num_minibatches
        
        if i % 100 == 0:
            print('Cost_J\t', epoch_cost)
        if i % 5 == 0:
            costs.append(epoch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }

    # lets save the parameters in a variable
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: X_set, Y: Y_set}, session=sess))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}, session=sess))
    
    return parameters

            
if __name__ == "__main__":
    pp = my_model(X_train, Y_train, X_test, Y_test) # num_epochs=10

    