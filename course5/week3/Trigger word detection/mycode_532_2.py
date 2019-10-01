### The code for the assignment 2 of course 5, week 3 ###

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, \
        TimeDistributed, LSTM, Conv1D, GRU, Bidirectional, \
        BatchNormalization, Reshape
from keras.optimizers import Adam


### Define some constants ###
Tx = 5511
n_freq = 101
Ty = 1375  # the length of labels


### The definition of this model ###
def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_inp = X = Input(shape = input_shape)
    
    X = Conv1D(filters = 196, strides = 4, kernel_size = 15)(X)
    X = TimeDistributed(BatchNormalization())(X)
    X = TimeDistributed(Activation('relu'))(X)
    X = TimeDistributed(Dropout(0.8))(X)

    X = GRU(units = 128, return_sequences = True)(X)
    X = TimeDistributed(Dropout(0.8))(X)
    X = TimeDistributed(BatchNormalization())(X)

    X = GRU(units = 128, return_sequences = True)(X)
    X = TimeDistributed(Dropout(0.8))(X)
    X = TimeDistributed(BatchNormalization())(X)
    X = TimeDistributed(Dropout(0.8))(X)

    X = TimeDistributed(Dense(1, activation = 'sigmoid'))(X) 

    return Model(inputs = X_inp, outputs = X)

model = model(input_shape = (Tx, n_freq))
model.summary()


### Load the pre-trained model ###
model = load_model('./models/tr_model.h5')


### Prepare to train this model ###
opt = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

model.fit(X, Y, batch_size = 5, epochs=1)

# Test the accuracy
loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)
 

### Make prediction ###
#   -- see the jupyter notebook




