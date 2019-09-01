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

X, Y, n_values, indices_values = load_music_utils()

from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import keras


n_a = 64
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation='softmax')
reshape_layer = Reshape((1,78))


def djmodel(Tx, n_a, n_values):
    # Initial the input values and output container.
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0') # a & c -> same dimension
    c0 = Input(shape=(n_a,), name='c0')
    outputs = []

    a = a0
    c = c0

    for t in range(Tx):
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshape_layer(x)
        a, _, c = LSTM_cell(x, initial_state=[a,c])
        y = densor(a)
        outputs.append(y)

    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


def train_djmodel():
    model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model.fit([X, a0, c0], list(Y), epochs=500)


def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    # Here, `LSTM_cell` is the trained model layer
    x0 = Input(shape=(1,n_values), name='x0')
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')

    a = a0
    c = c0
    x = x0
    outputs = []

    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a,c])
        y = densor(a)
        outputs.append(y)
        x = Lambda(one_hot)(y) # previous x => next y

    return Model(inputs=[x0, a0, c0], outputs=outputs)


def predict_inference():
    trained_model = music_inference_model(LSTM_cell, densor, n_values = n_values, n_a = n_a, Ty = 50)
    #
    out_stream = generate_music(trained_model)

    x0 = np.zeros((1,1,n_values))
    a0 = np.zeros((1,n_a))
    c0 = np.zeros((1,n_a))

    y = trained_model.predict([x0, a0, c0])
    indices = np.argmax(y, axis=-1)
    pred = keras.utils.to_categorical(indices, num_classes=78)

    print("np.argmax(results[12]) =", np.argmax(pred[12]))
    print("np.argmax(results[17]) =", np.argmax(pred[17]))
    print("list(indices[12:18]) =", list(indices[12:18]))
    return pred


train_djmodel()
predict_inference()
