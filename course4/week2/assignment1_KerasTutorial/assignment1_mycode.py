import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255. # (600, 64 ,64 ,3)
X_test = X_test_orig/255.
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T


def my_happy_model(x_shape):
    """
    || Input 
    =>(Padding > Conv > BatchNormal > Relu)
    => MaxPool
    => Full Connect Layer
    """
    X_input = Input(x_shape)
    X = X_input

    # X = ZeroPadding2D((3, 3), name = 'padding')(X_input)
    X = Conv2D(32, (7,7), padding = 'same', strides = (1,1), name = 'conv')(X)
    # Btach Normal 是对同一层的 Z 进行归一化
    X = BatchNormalization(axis = -1, name = 'batch_normal')(X)
    X = Activation('relu', name = 'activate')(X)

    X = MaxPooling2D((2,2), name = 'pooling')(X)
    X = Flatten()(X)
    X = Dense(1, activation = 'sigmoid', name = 'fc')(X)

    return Model(inputs = X_input, outputs = X, name ='Happy')


# Run Now
model_inst = my_happy_model(X_train.shape[1:])
model_inst.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_inst.fit(x=X_train, y=Y_train, batch_size=20, epochs=20)
# 试过 batch_size = 20, 12, 100
# 20, 100 : 前者收敛较快，但是最后到达的精度差不多。
# 12 : 收敛的不太稳定

test_result = model_inst.evaluate(x=X_test, y=Y_test)
print("Test Accuracy = ", test_result[1])


""" [Summary]
    1. Keras 的 API 很奇特，有些参数不是顺序参数，有些函数只接受命名参数，文档对此也说得很含糊。要对着文档看好。
    2. 很吃内存，而且遇到了（疑似）内存泄漏
    3. 对 BatchNormal 的 axis 参数设置的理解。BatchNorm 是对某一层的输出单元进行统计归一化。在CNN中，也就是说，是对输出的多个通道进行归一化。在深度神经网络中，归一化是对 [Z] = [z1, z2, ... , zn] 进行的。CNN中，要对输出的通道进行归一化，axis=3。
    4. 我之前一直错误地理解为对不同的样本进行归一化。这就是 Z^[l] 和 Z^(l) 的区别！
"""

