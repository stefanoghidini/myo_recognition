import tensorflow as tf

import numpy as np
import math
import os

from PIL import Image
import time

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, Dropout
from tensorflow.python.keras.layers import Reshape, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Activation

from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adadelta

from keras.utils import np_utils

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.client import device_lib
from keras.models import load_model

from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import itertools
import matplotlib.pyplot as plt

import scipy.io as sio

from data_augmentation import *
from labelling import *



#Uncomment the following line to use only CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Allow GPU to fill correctly all its memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if not tf.test.gpu_device_name():
    print("No GPU")
else:
    print("Yes GPU")

NAME = "Ste_Prova1_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('MyoConfusionMatrix')

# Loading Images
model_name = "5_Gest2_Ppl2"
train_list_data = np.load("src/Dataset/"+model_name+"/train_set.npy")
test_list_data = np.load("src/Dataset/"+model_name+"/test_set.npy")
val_list_data = np.load("src/Dataset/"+model_name+"/val_set.npy")

#Data augmentation
train_list_gaussian = np.reshape(gaussianNoise(train_list_data[0]), (1, 8, 15))
for i in range(1, train_list_data.shape[0]) :
    img_noised = np.reshape(gaussianNoise(train_list_data[i, :, :]), (1, 8, 15))
    train_list_gaussian = np.append(train_list_gaussian,img_noised,axis=0)

test_list_gaussian = np.reshape(gaussianNoise(test_list_data[0]), (1, 8, 15))
for i in range(1, test_list_data.shape[0]):
    img_noised = np.reshape(gaussianNoise(test_list_data[i, :, :]), (1, 8, 15))
    test_list_gaussian = np.append(test_list_gaussian, img_noised, axis=0)

val_list_gaussian = np.reshape(gaussianNoise(val_list_data[0]), (1, 8, 15))
for i in range(1, val_list_data.shape[0]):
    img_noised = np.reshape(gaussianNoise(val_list_data[i, :, :]), (1, 8, 15))
    val_list_gaussian = np.append(val_list_gaussian, img_noised, axis=0)

print(train_list_gaussian.shape)
print(test_list_gaussian.shape)
print(val_list_gaussian.shape)

# #Merge normal data with Gaussian noised data
# train_list = np.append(train_list_data, train_list_gaussian, axis=0)
# test_list = np.append(test_list_data, test_list_gaussian, axis=0)
# val_list = np.append(val_list_data, val_list_gaussian, axis=0)

#Merge normal data with Gaussian noised data
train_list = train_list_data
test_list = test_list_data
val_list = val_list_data

print(train_list.shape)
print(test_list.shape)
print(val_list.shape)

flat = np.reshape(train_list[0,:,:].flatten(),(1,120))
matrix_train = np.array(flat, dtype="int32")
flat = np.reshape(test_list[0,:,:].flatten(),(1,120))
matrix_test = np.array(flat, dtype="int32")
flat = np.reshape(val_list[0, :, :].flatten(), (1, 120))
matrix_val = np.array(flat, dtype="int32")

for i in range(1,train_list.shape[0]) :
    #Flat and reshape train
    flat = np.reshape(train_list[i, :, :].flatten(), (1, 120))
    matrix_train = np.append(matrix_train, flat, axis=0)

for i in range(1, test_list.shape[0]):
    #Flat and reshape test
    flat = np.reshape(test_list[i, :, :].flatten(), (1, 120))
    matrix_test = np.append(matrix_test, flat, axis=0)

for i in range(1, val_list.shape[0]):
    #Flat and reshape val
    flat = np.reshape(val_list[i, :, :].flatten(), (1, 120))
    matrix_val = np.append(matrix_val, flat, axis=0)

print(matrix_train.shape)
print(matrix_test.shape)
print(matrix_val.shape)

# Labeling
num_classes = 5

label_train_data = labelling(train_list_data.shape[0], num_classes)
label_test_data = labelling(test_list_data.shape[0], num_classes)
label_val_data = labelling(val_list_data.shape[0], num_classes)

label_train_gaussian = labelling(train_list_gaussian.shape[0], num_classes)
label_test_gaussian = labelling(test_list_gaussian.shape[0], num_classes)
label_val_gaussian = labelling(val_list_gaussian.shape[0], num_classes)

# label_train = np.append(label_train_data, label_train_gaussian)
# label_test = np.append(label_test_data, label_test_gaussian)
# label_val = np.append(label_val_data, label_val_gaussian)

# label_train = label_train_data
# label_test = label_test_data
# label_val = label_val_data

# print(label_test)
# print(label_test.shape)

label_train = np.load("src/Dataset/" + model_name + "/train_labels.npy")
label_test = np.load("src/Dataset/" + model_name + "/test_labels.npy")
label_val = np.load("src/Dataset/" + model_name + "/val_labels.npy")

print(label_train)
print(label_test)
print(label_val)

# label_train = np.ones((train_list.shape[0], ), dtype=int)
# label_train[0      : 147] = 0     #exercise 1
# label_train[147  : 294] = 1     #exercise 2
# label_train[294  : 441] = 2     #exercise 3
# label_train[441:train_list.shape[0]] = 3  #exercise 4

# label_test = np.ones((test_list_data.shape[0], ), dtype=int)
# label_test[0      :   50] = 0   #exercise 1
# label_test[50   :   100] = 1   #exercise 2
# label_test[100   :  150] = 2   #exercise 3
# label_test[150:test_list_data.shape[0]] = 3  #exercise 4

# label_val = np.ones((val_list.shape[0], ), dtype=int)
# label_val[0  :  48] = 0   #exercise 1
# label_val[48 :   96] = 1   #exercise 2
# label_val[96  : 144] = 2   #exercise 3
# label_val[144:val_list.shape[0]] = 3  #exercise 4

# print(label_train.shape)
# print(label_test.shape)
# print(label_val.shape)


# Training set.
X_train, y_train = shuffle(matrix_train, label_train, random_state = 3)
print(X_train.shape)
print(y_train.shape)

# .. to images again! (For convolution)
img_rows = 8
img_cols = 15


print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
print(X_train.shape)
# X_train = X_train.reshape
X_train = np.transpose(X_train, (0,2,3,1))

print(X_train.shape)
# .. categorical labeling


Y_train= np_utils.to_categorical(y_train, num_classes)

# Test set.
X_test = matrix_test.reshape(matrix_test.shape[0], 1, img_rows, img_cols)
X_test = np.transpose(X_test, (0,2,3,1))
# .. categorical labeling

Y_test= np_utils.to_categorical(label_test, num_classes)
# Validation set.

X_val = matrix_val.reshape(matrix_val.shape[0], 1, img_rows, img_cols)

X_val = np.transpose(X_val, (0,2,3,1))
# .. categorical labeling

Y_val= np_utils.to_categorical(label_val, num_classes)

# Model
model = Sequential()

# Stage 1
model.add(Conv2D(kernel_size=(3,4), strides=1, filters=32, padding='same',data_format = 'channels_last', name='layer_conv1', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))

# Stage 2
model.add(Conv2D(kernel_size=(3,3), strides=1, filters=32, padding='same', name='layer_conv2'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = 2, strides=1))

# Stage 3
model.add(Conv2D(kernel_size=(2,1), strides=1, filters=32, padding='same', name='layer_conv4'))
model.add(Activation('relu'))

# Stage 4
model.add(Conv2D(kernel_size=(1,3), strides=(1,2), filters=64, padding='same', name='layer_conv5'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(pool_size = 2, strides=(2,2)))

# Stage 5
model.add(Conv2D(kernel_size=(1,2), strides=1, filters=64, padding='same', name='layer_conv7'))
model.add(Activation('relu'))

# Stage 6
model.add(Conv2D(kernel_size=(2,2), strides=1, filters=128, padding='same', name='layer_conv8'))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Flatten())

# Stage 7
model.add(Dense(units = 512))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Stage 8
model.add(Dense(units = 128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Stage 9
model.add(Dense(units = 64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(5)) #num_classes
model.add(Activation('softmax'))

model.summary()

# Optimizer
# sgd = SGD(lr=0.01, decay=1e-6,  momentum=0.9, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Fit
model.fit(x = X_train, y = Y_train, validation_data=(X_val, Y_val), epochs = 500, batch_size = 256, verbose = 1, callbacks = [tensorboard])

# Evaluation
result = model.evaluate(x=X_test,y=Y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)

model.save("src/Models/Myo_Model_" + model_name + ".h5")

# Confusion Matrix
rounded_predictions = model.predict_classes(X_test, batch_size = 1, verbose = 0)
conf_matrix = confusion_matrix(label_test, rounded_predictions)

cm_plot_labels = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5']

plot_confusion_matrix(conf_matrix, cm_plot_labels, title = 'Confusion Matrix')
