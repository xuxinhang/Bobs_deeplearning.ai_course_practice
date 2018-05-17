import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'


def pad_with_const(X, pad, const = 0): # check pass
    return np.pad(X, pad, 'constant', constant_values = const)

def test_for_pad_with_const():
    np.random.seed(123)
    raw = np.random.randn(2,4,4)
    aft = pad_with_const(raw, 2)

    fig, subplots = plt.subplots(2,1)
    subplots[0].set_title('Raw Input')
    subplots[0].imshow(raw[0,:,:])
    subplots[1].set_title('Padded Output')
    subplots[1].imshow(aft[2,:,:])
    plt.show(fig)


# Test the result of pad_with_const
# test_for_pad_with_const()

def conv_single_step(A_sub, W, b): # Check Pass
    rst = np.sum(np.multiply(A_sub, W))
    rst += b
    return rst

def apply_conv_filter(A, W, b, params, acti_func = 0):
    """
    A: The whole input for this layer, not a slice
    W: Single filter
    b：(single number, instead of a vector)
    params = {pad: , stride: }
    """
    pad = params['pad']
    stride = params['stride']
    # NOTICE | A_shape here is the shape before padding 
    A_shape = A.shape 
    W_shape = W.shape
    b = float(b)

    A = pad_with_const(A, ((pad, pad), (pad, pad), (0,0)))
    A_h, A_w, A_c = A.shape
    W_h, W_w, W_c = W.shape

    Z_w = (A_w - W_w) // stride + 1
    Z_h = (A_h - W_h) // stride + 1
    Z = np.zeros((Z_h, Z_w))

    # j = row index ; i = column index
    for j in range(0, A_h-W_h+1, stride):
        for i in range(0, A_w-W_w+1, stride):
            a_slice = A[j: j+W_h, i: i+W_w]
            rst = np.sum(np.multiply(a_slice, W)) + b
            Z[j//stride, i//stride] = rst
            
    return Z


def test_apply_conv_filter (): # Check Pass
    A = np.array([[1,2,3, 4], [4,3,2,1], [5,6,7,8], [8,7,6,5]])
    W = np.array([[1,0], [0,1]])
    b = .2
    rr = apply_conv_filter(A, W, b, {"stride": 2, "pad": 1})
    print(rr)


def conv_forward(A, W, b, params):
    pad = params['pad']
    stride = params['stride']
    (m, A_h, A_w, A_c) = A.shape
    (f, f, W_nc, W_outnc) = W.shape
    (n_h, n_w, n_c) = (int((A_h+2*pad-f)/stride+1), int((A_w+2*pad-f)/stride+1), W_outnc)

    Z = np.zeros((m, n_h, n_w, n_c))

    for i in range(m):
        for c in range(n_c):
            Z[i, ..., c] = apply_conv_filter(A[i], W[...,c], b[..., c], params)

    cache = (A, W, b, params)
    return Z, cache


def test_forward_conv_layer(): # Pass Check
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
                "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


def pool_forward(A, params, mode = "max"):
    m, A_h, A_w, A_c = A.shape
    f = params['f']
    stride = params['stride']

    # Output shape
    Z_h = (A_h - f)//stride + 1
    Z_w = (A_w - f)//stride + 1
    Z_c = A_c
    Z = np.zeros((m, Z_h, Z_w, Z_c))

    for j in range(0, Z_h):
        for i in range(0, Z_w):
            ia = i * stride
            ja = j * stride
            # different modes
            if mode == 'average':
                rst = np.average(A[:,ja:ja+f, ia:ia+f,:], axis=(1,2))
            else:
                rst = np.max(A[:,ja:ja+f, ia:ia+f,:], axis=(1,2))
            
            Z[:,j,i,:] = rst

    cache = (A, params)
    return Z, cache

def test_pool_forward(): # Check pass (Wow! I should use the numpy's vector method!)
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride" : 2, "f": 3}

    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode = "average")
    print("mode = average")
    print("A =", A)


def conv_backward(dZ, cache):
    A, W, b, params = cache
    m, A_h, A_w, A_c = A.shape
    f, f, W_c, W_oc = W.shape
    m, Z_h, Z_w, Z_c = dZ.shape

    pad = params['pad']
    stride = params['stride']

    dA = np.zeros((m, A_h, A_w, A_c))
    dW = np.zeros((f, f, W_c, W_oc))
    db = np.zeros((1, 1, 1, Z_c))

    dAp = pad_with_const(dA, ((0,0), (pad,pad), (pad,pad), (0,0)))
    Ap  = pad_with_const(A, ((0,0), (pad,pad), (pad,pad), (0,0)))

    for i in range(m):
        for c in range(Z_c):
            for h in range(Z_h):
                for w in range(Z_w):
                    a_h_start = stride * h
                    a_h_end = a_h_start + f
                    a_w_start = stride * w
                    a_w_end = a_w_start + f

                    dAp[i, a_h_start:a_h_end, a_w_start:a_w_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += Ap[i, a_h_start:a_h_end, a_w_start:a_w_end, :] * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]

        dA[i,:,:,:] = dAp[i, pad:-pad, pad:-pad, :]

    return dA, dW, db

def test_conv_backward(): # Check pass
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
                "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

    np.random.seed(1)
    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

def create_mask_from_window(x):
    return (x == np.max(x))

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = n_H * n_W
    a = np.ones(shape) * dz / average
    return a

def pool_backward(dZ, cache, mode = 'max'):
    A, params = cache
    f = params['f']
    stride = params['stride']

    m, A_h, A_w, A_c = A.shape
    m, Z_h, Z_w, Z_c = dZ.shape
    dA = np.zeros(A.shape)
    
    for i in range(m):
        for c in range(Z_c):
            for h in range(Z_h):
                for w in range(Z_w):
                    a_h_start = stride * h
                    a_h_end = a_h_start + f
                    a_w_start = stride * w                    
                    a_w_end = a_w_start + f

                    if mode == 'average':
                        dA[i, a_h_start:a_h_end, a_w_start:a_w_end, c] += distribute_value(dZ[i,h,w,c], (f,f))
                    else:
                        mask = create_mask_from_window(A[i, a_h_start:a_h_end, a_w_start:a_w_end, c])
                        dA[i, a_h_start:a_h_end, a_w_start:a_w_end, c] += mask * dZ[i,h,w,c]

    return dA

def test_pool_backwrad():
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride" : 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])  
    print()
    dA_prev = pool_backward(dA, cache, mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1]) 

test_pool_backwrad()

print("END HERE")
    














