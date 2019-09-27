from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
# %matplotlib inline


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    s_prev_repeated = repeator(s_prev)
    concat = concatenator([s_prev_repeated, a])
    # e_pre = densor2(densor1(concat))  # Why to use two densor?
    e = densor(concat)
    alpha = activator(e)
    context = dotor([alpha, a])

    return context


n_a = 64  # 32
n_s = 128 # 64
# post_activation_LSTM_cell = LSTM(n_s, return_state = True)
# output_layer = Dense(len(machine_vocab), activation=softmax)

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    Y = []

    pre_LSTM_layer = Bidirectional(LSTM(units=n_a, return_sequences=True))
    post_LSTM_cell = LSTM(n_s, return_state=True)
    output_densor = Dense(machine_vocab_size, activation=softmax)

    X = Input(shape=(Tx, human_vocab_size))
    a = pre_LSTM_layer(X)

    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_LSTM_cell(context, initial_state=[s, c])
        y = output_densor(s)
        Y.append(y)

    return Model(input=[X, s0, c0], output=Y)


# Initialize Model
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()
'''
opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt ,loss='categorical_crossentropy', metrics=['accuracy'])

# Train this Model
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
'''


# Train this Model
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01) # lr = learning_rate
model.compile(optimizer=opt ,loss='categorical_crossentropy', metrics=['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh, s0, c0], outputs, epochs=55, batch_size=100)


# Load exsiting weight data
model.load_weights('models/model.h5')


# Test
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:

    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = np.array([source])
    # print(source.shape)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))

attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "8 Aug 2010", num = 6, n_s = 128)
