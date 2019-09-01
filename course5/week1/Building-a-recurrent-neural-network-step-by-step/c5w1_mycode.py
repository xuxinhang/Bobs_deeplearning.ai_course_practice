import numpy as np
from rnn_utils import *

# Basic RNN Cell

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    return a_next, yt_pred, (a_next, a_prev, xt, parameters)

def test_rnn_cell_forward():
    np.random.seed(1)
    xt = np.random.randn(3,10)
    a_prev = np.random.randn(5,10)
    Waa = np.random.randn(5,5)
    Wax = np.random.randn(5,3)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    print("a_next[4] = ", a_next[4])
    print("a_next.shape = ", a_next.shape)
    print("yt_pred[1] =", yt_pred[1])
    print("yt_pred.shape = ", yt_pred.shape)

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """

    # m represents the number of samples
    _, m, T_x = np.shape(x)
    n_y, _ = np.shape(parameters['Wya'])
    n_a, _ = np.shape(parameters['Wax'])

    a_prev = a0

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    list_of_caches = []

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_prev, parameters)
        y_pred[:,:,t] = yt_pred
        a[:,:,t] = a_next
        list_of_caches.append(cache)
        a_prev = a_next

    return a, y_pred, (list_of_caches, x)

def test_rnn_forward():
    np.random.seed(1)
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Waa = np.random.randn(5,5)
    Wax = np.random.randn(5,3)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a, y_pred, caches = rnn_forward(x, a0, parameters)
    print("a[4][1] = ", a[4][1])
    print("a.shape = ", a.shape)
    print("y_pred[1][3] =", y_pred[1][3])
    print("y_pred.shape = ", y_pred.shape)
    print("caches[1][1][3] =", caches[1][1][3])
    print("len(caches) = ", len(caches))

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    Wc = parameters['Wc']
    bc = parameters['bc']
    Wy = parameters['Wy']
    by = parameters['by']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wf = parameters['Wf']
    bf = parameters['bf']

    concat = np.concatenate((a_prev, xt))

    it = sigmoid(np.dot(Wi, concat) + bi)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    ot = sigmoid(np.dot(Wo, concat) + bo)

    cct = np.tanh(np.dot(Wc, concat) + bc)
    ct = it * cct + ft * c_prev
    at = ot * np.tanh(ct)
    yt_pred = softmax(np.dot(Wy, at) + by)

    return at, ct, yt_pred, (at, ct, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    

def test_lstm_cell_forward():
    np.random.seed(1)
    xt = np.random.randn(3,10)
    a_prev = np.random.randn(5,10)
    c_prev = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)
    Wy = np.random.randn(2,5)
    by = np.random.randn(2,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
    print("a_next[4] = ", a_next[4])
    print("a_next.shape = ", c_next.shape)
    print("c_next[2] = ", c_next[2])
    print("c_next.shape = ", c_next.shape)
    print("yt[1] =", yt[1])
    print("yt.shape = ", yt.shape)
    print("cache[1][3] =", cache[1][3])
    print("len(cache) = ", len(cache))


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    _, m, T_x = np.shape(x)
    n_c, _ = np.shape(parameters['Wc'])
    n_y, _ = np.shape(parameters['Wy'])
    n_a = n_c

    a_prev = a0
    c_prev = np.zeros((n_c, m))
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_c, m, T_x))
    y = np.zeros((n_y, m, T_x))
    list_of_caches = []

    for t in range(T_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t], a_prev, c_prev, parameters)
        a[:,:,t] = a_next
        c[:,:,t] = c_next
        y[:,:,t] = yt_pred
        list_of_caches.append(cache)
        a_prev = a_next
        c_prev = c_next

    return a, y, c, (list_of_caches, x)

def test_lstm_forward():
    np.random.seed(1)
    x = np.random.randn(3,10,7)
    a0 = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)
    Wy = np.random.randn(2,5)
    by = np.random.randn(2,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)
    print("a[4][3][6] = ", a[4][3][6])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1])
    print("len(caches) = ", len(caches))



# Backforward spread

def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """

    a_next, a_prev, xt, parameters = cache
    Waa = parameters['Waa']
    Wax = parameters['Wax']

    temp = da_next * (1 - a_next * a_next)

    da_prev = np.dot(Waa.T, temp)
    dxt = np.dot(Wax.T, temp)
    dWaa = np.dot(temp, a_prev.T)
    dWax = np.dot(temp, xt.T)
    dba = np.sum(temp, keepdims=True, axis=1)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients

def test_rnn_cell_backward():
    np.random.seed(1) # HACK
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Waa = np.random.randn(5,5)
    Wax = np.random.randn(5,3)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)

    np.random.seed(1)
    xt = np.random.randn(3,10)
    a_prev = np.random.randn(5,10)
    Wax = np.random.randn(5,3)
    Waa = np.random.randn(5,5)
    Wya = np.random.randn(2,5)
    b = np.random.randn(5,1)
    by = np.random.randn(2,1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

    da_next = np.random.randn(5,10)
    gradients = rnn_cell_backward(da_next, cache)
    print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
    print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
    print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
    print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dba\"].shape =", gradients["dba"].shape)


def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """

    list_of_caches, x = caches
    n_a, m, T_x = np.shape(da)
    n_x, m, T_x = np.shape(x)

    dx = np.zeros(np.shape(x))
    da0 = np.zeros((n_a, m))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))

    da_next = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t] + da_next, list_of_caches[t])
        dx[:,:,t] = gradients['dxt']
        dWax += gradients['dWax']
        dWaa += gradients['dWaa']
        dba += gradients['dba']
        da_next = gradients['da_prev']

    da0 = da_next
    return { 'dx': dx, 'da0': da0, 'dWax': dWax, 'dWaa': dWaa, 'dba': dba }

def test_rnn_backward():
    np.random.seed(1)
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Wax = np.random.randn(5,3)
    Waa = np.random.randn(5,5)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    a, y, caches = rnn_forward(x, a0, parameters)
    da = np.random.randn(5, 10, 4)
    gradients = rnn_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dba\"].shape =", gradients["dba"].shape)


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    Wf = parameters['Wf']
    Wo = parameters['Wo']
    Wi = parameters['Wi']
    Wc = parameters['Wc']
    n_a, _ = np.shape(a_next)
    
    dot = da_next * np.tanh(c_next)
    dct = da_next * ot * (1 - np.square(np.tanh(c_next)))
    dcct = dc_next * it + dct * it
    dit = dc_next * cct + dct * cct
    dft = dc_next * c_prev + dct * c_prev
    
    temp = np.concatenate((a_prev, xt)).T
    dWf =  np.dot(dft * ft * (1-ft), temp)
    dWo =  np.dot(dot * ot * (1-ot), temp)
    dWi =  np.dot(dit * it * (1-it), temp)
    dWc =  np.dot(dcct * (1 - np.square(cct)), temp)

    dbf = np.sum(dft * ft * (1-ft), axis=1, keepdims=True)
    dbi = np.sum(dit * it * (1-it), axis=1, keepdims=True)
    dbo = np.sum(dot * ot * (1-ot), axis=1, keepdims=True)
    dbc = np.sum(dcct * (1 - np.square(cct)), axis=1, keepdims=True)

    dc_prev = dc_next * ft + dct * ft
    da_prev_xt = Wc.T @ (dcct * (1-np.square(cct))) \
        + Wi.T @ (dit * it * (1-it)) \
        + Wo.T @ (dot * ot * (1-ot)) \
        + Wf.T @ (dft * ft * (1-ft))
    da_prev = da_prev_xt[:n_a,:]
    dxt = da_prev_xt[n_a:,:]

    return {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev,"dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

def test_lstm_cell_backward():
    np.random.seed(1)
    xt = np.random.randn(3,10)
    a_prev = np.random.randn(5,10)
    c_prev = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)
    Wy = np.random.randn(2,5)
    by = np.random.randn(2,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    da_next = np.random.randn(5,10)
    dc_next = np.random.randn(5,10)
    gradients = lstm_cell_backward(da_next, dc_next, cache)
    print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
    print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
    print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
    print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
    print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
    print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)


def lstm_backward(da, caches):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    list_of_caches, x = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = list_of_caches[0]
    n_a, m, T_x = np.shape(da)
    n_x, m = np.shape(x1)

    dc_next = np.zeros(np.shape(da[:,:,0]))
    da_next = np.zeros(np.shape(da[:,:,0]))

    dx = np.zeros((n_x, m, T_x))
    W_shape = (n_a, n_a + n_x)
    b_shape = (n_a, 1)
    dWf = np.zeros(W_shape)
    dWi = np.zeros(W_shape)
    dWo = np.zeros(W_shape)
    dWc = np.zeros(W_shape)
    dbf = np.zeros(b_shape)
    dbi = np.zeros(b_shape)
    dbo = np.zeros(b_shape)
    dbc = np.zeros(b_shape)

    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:,:,t] + da_next, dc_next, list_of_caches[t])
        # da_next = gradients['da_prev']
        # dc_next = gradients['dc_prev']
        dx[:,:,t] = gradients['dxt']
        dWc += gradients['dWc']
        dWi += gradients['dWi']
        dWf += gradients['dWf']
        dWo += gradients['dWo']
        dbc += gradients['dbc']
        dbi += gradients['dbi']
        dbf += gradients['dbf']
        dbo += gradients['dbo']

    da0 = gradients['da_prev']

    return {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
            "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

def test_lstm_backward():
    np.random.seed(1) # HACK
    xt = np.random.randn(3,10)
    a_prev = np.random.randn(5,10)
    c_prev = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)
    Wy = np.random.randn(2,5)
    by = np.random.randn(2,1)
    np.random.seed(1)
    x = np.random.randn(3,10,7)
    a0 = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)

    da = np.random.randn(5, 10, 4)
    gradients = lstm_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

test_lstm_backward()
