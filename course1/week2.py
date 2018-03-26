import numpy as np
#import matplotlib.pyplot as pyplot
#import matplotlib as mat
#from mat import pyplot as plt
import h5py
import scipy
#from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# %matplotlib inline

def sigma (x):
  return 1 / (1 + np.exp(-x))

# 导入数据集 Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_number = train_set_x_orig.shape[0]
# 向量化训练数据
train_set_x_flat = train_set_x_orig.reshape(train_set_number, -1).T
X = train_set_x_flat/255
Y = train_set_y

# 初始化参数
W = np.random.rand(X.shape[0], 1) * 0.01
b = 0

# 开始计算
alaph = 0.009
for i in range(9000):
  A = sigma(np.dot(W.T, X) + b) # 计算值
  J = 1/train_set_number *  np.sum(-Y*np.log(A) -(1-Y)*np.log(1-A))
  # 反向传播
  dW = 1/train_set_number * np.dot(X, (A-Y).T)
  db = 1/train_set_number * np.sum(A-Y)
  # 梯度下降
  W -= alaph * dW
  b -= alaph * db
  # 打印结果
  if i % 100 == 0:
    print('dW', dW, 'db', db)
    print('W', W, 'b', b)
    print("Loss:J", J)

# 输出结果
print('Now, result out')
print('W', W, 'b', b)

### 校验结果,使用训练数据 
Y_hat = np.zeros((1, X.shape[1]))
# 开始计算
A = np.dot(W.T, X) + b
for i in range(A.shape[1]):
  Y_hat[0][i] = A[0][i] > 0.5
# 校验
fail_counter = np.sum(np.absolute(Y_hat-train_set_y))
print("Fail rate : ", fail_counter/train_set_number)


### 校验结果,使用测试数据 
# 扁平化
test_set_number = test_set_x_orig.shape[0]
test_set_x = test_set_x_orig.reshape(test_set_number, -1).T / 255
Y_hat = np.zeros((1, test_set_number))
# 开始计算
A = np.dot(W.T, test_set_x) + b
for i in range(A.shape[1]):
  Y_hat[0][i] = A[0][i] > 0.5
# 校验
fail_counter = np.sum(np.absolute(Y_hat-test_set_y))
print("Fail rate : ", fail_counter / test_set_number)




## PUT YOUR IMAGE NAME) 
num_px =64
imgs = ["dog.jpg", "dog2.jpg", "dog3.jpg", "cat.jpg", "cat2.jpg", "cat3.jpg", "cat4.jpg"]
for my_image in imgs:
    
    # my_image = "dog.jpg"   # change this to the name of your image file 
    ## END CODE HERE ##

    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = sigma( np.dot(W.T, my_image) + b )

    #plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your::" + my_image)

  










