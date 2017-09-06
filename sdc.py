import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math



def change_bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    rand = random.uniform(0.5,1.)
    hsv[:,:,2] = rand*hsv[:,:,2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
   # plt.imshow(new_img)
   # plt.show()
    return new_img

def crop_sky(img):
    #120*320
    #196*455 sulle reali
    crop_img = img[60::, ::]
   # plt.imshow(crop_img)
   # plt.show()
    return crop_img


data_path="/home/norman/Desktop/driving_dataset/data.txt"

img_paths=[]
steers=[]
with open(data_path) as file:
    for line in file:
        if line.split(',')[0] == "center": continue
        img_paths.append("/home/norman/Desktop/driving_dataset/" + line.split(' ')[0])
        steers.append(line.split(' ')[1].strip())


def gen_batch(batch_size):
    batch_x=np.zeros((batch_size,196,455,3))
    batch_y=np.zeros((batch_size,1))
    pointer=0
    (im_paths, steerss)=shuffle(img_paths, steers)
    while True:
        for i in range(batch_size):
            img=plt.imread(im_paths[pointer])
            steer=steerss[pointer]
            new_img=crop_sky(change_bright(img))
            
            batch_x[i]=new_img
            batch_y[i]=steer
            pointer+=1
            if pointer==len(im_paths)-1: pointer=0
        
        yield batch_x, batch_y


generator=gen_batch(5)
input_shape = (196,455,3)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), kernel_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, kernel_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, kernel_regularizer = l2(0.001)))
model.add(Dense(1, kernel_regularizer = l2(0.001)))
adam = Adam(lr = 0.0001)
model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
model.summary()


model.fit_generator(generator, samples_per_epoch = int(len(img_paths)/5-10), nb_epoch=3)



