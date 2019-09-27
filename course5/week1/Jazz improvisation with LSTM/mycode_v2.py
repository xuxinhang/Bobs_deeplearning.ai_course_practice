from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


def djmodel(Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """

    X = Input(shape=(1, n_values))
    X = LSTM(n_a, input_length=Tx, return_sequences=True)
    Y = Dense(n_values, activation='softmax')

    model = Model(input=X, output=Y)
    return model


def generate_sample_model(LSTM_layer, dense_layer, n_values = 78, n_a = 64, Ty = 100):
    LSTM_cell = LSTM(n_a, return_state = True)
    LSTM_cell.set_weights(LSTM_layer.get_weights())
    densor = Dense(n_values, activation='softmax')
    densor.set_weights(dense_layer.get_weights())

    x = x0 = Input(shape=(1, n_values), name='x0')
    a = a0 = Input(shape=(n_a, ), name='a0')
    c = c0 = Input(shape=(n_a, ), name='c0')

    outputs = []

    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        y = densor(a)
        outputs.append(y)
        x = Lambda(one_hot)(y)

    smaple_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return sample_model

sample_model = generate_smaple_model()
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
sample_model.predict([x_initializer, a_initializer, c_initializer])



