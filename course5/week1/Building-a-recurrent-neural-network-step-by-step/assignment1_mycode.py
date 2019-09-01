import numpy as np
from rnn_utils import *

def rnn_cell_forward(xt, a_prev, params): # Test Pass
    W_aa = params['Waa']
    W_ax = params['Wax']
    W_ya = params['Wya']
    b_a =  params['ba']
    b_y =  params['by']

    a_next = np.tanh(np.dot(W_aa, a_prev) + np.dot(W_ax, xt) + b_a)
    yt = softmax(np.dot(W_ya, a_next) + b_y)
    cache = (a_next, a_prev, xt, params)

    return a_next, yt, cache

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


def rnn_forward(x, a_0, params):  # Pass Test
    caches = []  # cahces for each loop
    n_x, m, T_x = x.shape
    n_y, n_a = params['Wya'].shape
    y = np.zeros((n_y, m, T_x))
    a = np.zeros((n_a, m, T_x))

    a_temp = a_0
    for t in range(T_x):
        a_temp, y[:,:,t], cache = rnn_cell_forward(x[:,:,t], a_temp, params)
        caches.append(cache)
        a[:,:,t] = a_temp

    caches = (caches, x)
    return a, y, caches

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


def lstm_cell_forward(xt, a_prev, c_prev, params): # Pass Test
    W_f = params["Wf"]
    b_f = params["bf"]
    W_u = params["Wi"]
    b_u = params["bi"]
    W_c = params["Wc"]
    b_c = params["bc"]
    W_o = params["Wo"]
    b_o = params["bo"]
    W_y = params["Wy"]
    b_y = params["by"]

    concat = np.concatenate((a_prev, xt))

    ga_f = sigmoid(W_f @ concat + b_f)
    ga_u = sigmoid(W_u @ concat + b_u) ###
    c_tilde = np.tanh(W_c @ concat + b_c)
    c = ga_f * c_prev + ga_u * c_tilde
    ga_o = sigmoid(W_o @ concat + b_o)
    a = ga_o * np.tanh(c)
    y = softmax(W_y @ a + b_y)

    cache = (a, c, a_prev, c_prev, ga_f, ga_u, c_tilde, ga_o, xt, params)
    return a, c, y, cache

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


def lstm_forward(x, a_0, params):  # Test Pass
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_ax = params['Wy'].shape
    n_c, n_ax = params['Wc'].shape
    y = np.zeros((n_y, m, T_x))
    c = np.zeros((n_c, m, T_x))
    a = np.zeros((n_c, m, T_x)) # Here, `n_c` is equal to `n_a`

    a_t = a_0
    c_t = np.zeros((n_c, m))

    for t in range(T_x):
        a_t, c_t, y_t, cache_t = lstm_cell_forward(x[:,:,t], a_t, c_t, params)
        y[:,:,t] = y_t
        a[:,:,t] = a_t
        c[:,:,t] = c_t
        caches.append(cache_t)

    caches = (caches, x)
    return a, y, c, caches

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


def rnn_cell_backward(da_next, cache):
    a_next, a_prev, xt, params = cache
    W_aa = params['Waa']
    W_ax = params['Wax']
    W_ya = params['Wya']
    b_a = params['ba']
    b_y = params['by']
    n_x, m = xt.shape

    dz = (1 - a_next ** 2) * da_next
    dW_aa = (a_prev @ dz.T).T  # 如何推导？
    dW_ax = (xt @ dz.T).T
    db_a  = np.sum(dz, axis=-1, keepdims=True) # / m
    dxt   = W_ax.T @ dz
    da_prev=W_aa.T @ dz 

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dW_ax, "dWaa": dW_aa, "dba": db_a}
    return gradients

def test_rnn_cell_backward():
    np.random.seed(1)
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Waa = np.random.randn(5,5)
    Wax = np.random.randn(5,3)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)  # Important!

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
    (caches, x) = caches
    (a_1, a_0, x_1, params) = caches[0]

    n_a, n_x = params['Wax'].shape
    n_x, m, T_x = x.shape

    dW_aa = np.zeros((n_a, n_a))
    dW_ax = np.zeros((n_a, n_x))
    da_0  = np.zeros((n_a, m))
    dx    = np.zeros((n_x, m, T_x))
    db_a  = np.zeros((n_a, 1))
    da_prev=np.zeros((n_a, m))

    for t in range(T_x-1, -1, -1):
        grads = rnn_cell_backward(da[:, :, t] + da_prev, caches[t]) # Notice..
        dW_aa += grads['dWaa']
        dW_ax += grads['dWax']
        db_a  += grads['dba']
        dx[:,:,t] = grads['dxt']
        da_prev = grads['da_prev']

    da_0 = da_prev
    gradients = {"dx": dx, "da0": da_0, "dWax": dW_ax, "dWaa": dW_aa,"dba": db_a}
    return gradients

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
    
    a_next, c_next, a_prev, c_prev, ga_f, ga_u, c_tilde, ga_o, xt, params = cache
    W_f = params['Wf']
    W_u = params["Wi"]
    W_c = params["Wc"]
    W_o = params["Wo"]
    n_a, m = a_next.shape

    dga_o = da_next * np.tanh(c_next) * ga_o * (1 - ga_o) #
    dc_tilde = (dc_next * ga_u + ga_o * (1-np.tanh(c_next)**2) * ga_u * da_next) * (1 - c_tilde ** 2)
    dga_u = (dc_next * c_tilde + ga_o * (1-np.tanh(c_next)**2) * c_tilde * da_next) * ga_u * (1-ga_u)
    dga_f = (dc_next * c_prev + ga_o * (1-np.tanh(c_next)**2) * c_prev * da_next) * ga_f * (1-ga_f)

    concat = np.concatenate((a_prev, xt), axis=0)
    dW_f = dga_f @ concat.T
    dW_u = dga_u @ concat.T
    dW_c = dc_tilde @ concat.T
    dW_o = dga_o @ concat.T #

    dc_prev = dc_next * ga_f + ga_o * (1-np.tanh(c_next)**2) * ga_f * da_next
    dxt = W_f[:, n_a:].T @ dga_f + W_u[:, n_a:].T @ dga_u + W_c[:, n_a:].T @ dc_tilde + W_o[:, n_a:].T @ dga_o
    da_prev = W_f[:, :n_a].T @ dga_f + W_u[:, :n_a].T @ dga_u + W_c[:, :n_a].T @ dc_tilde + W_o[:, :n_a].T @ dga_o
    
    db_f = np.sum(dga_f, axis=1, keepdims=True) #
    db_u = np.sum(dga_u, axis=1, keepdims=True)
    db_c = np.sum(dc_tilde, axis=1, keepdims=True)
    db_o = np.sum(dga_o, axis=1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dW_f,"dbf": db_f, "dWi": dW_u,"dbi": db_u,
                "dWc": dW_c,"dbc": db_c, "dWo": dW_o,"dbo": db_o}
    return gradients

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

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])
    
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:,:,t],dc_prevt,caches[t])
        # da_prevt, dc_prevt = gradients['da_prev'], gradients["dc_prev"]
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:,:,t] = gradients['dxt']
        dWf = dWf+gradients['dWf']
        dWi = dWi+gradients['dWi']
        dWc = dWc+gradients['dWc']
        dWo = dWo+gradients['dWo']
        dbf = dbf+gradients['dbf']
        dbi = dbi+gradients['dbi']
        dbc = dbc+gradients['dbc']
        dbo = dbo+gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']
    
    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients


    caches, x = caches
    n_x, m, T_x = x.shape
    n_a, m, T_x = da.shape

    da0 = np.zeros([n_a, m])
    da_prev = np.zeros((n_a, m))
    dc_prev = np.zeros((n_a, m))
    dx = np.zeros((n_x, m, T_x))
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])

    for t in range(T_x-1, -1, -1):
        grads = lstm_cell_backward(da_prev + da[:,:,t], dc_prev, caches[t])
        dx[:,:,t] = grads['dxt']
        dWf += grads['dWf']
        dWi += grads['dWi']
        dWc += grads['dWc']
        dWo += grads['dWo']
        dbf += grads['dbf']
        dbi += grads['dbi']
        dbc += grads['dbc']
        dbo += grads['dbo']

    da0 = grads['da_prev']
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
            "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    return gradients

def test_lstm_backward():
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