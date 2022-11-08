import tensorflow as tf
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam

#constants
in_shape = (64,64,1)
out_shape = (64,64,2)

#models
def generator():
  input = Input(shape=(in_shape))

  conv1 = Conv2D(64,kernel_size=1,strides=1)(input)
  conv1 = LeakyReLU(0.2)(conv1)

  conv2 = Conv2D(128,kernel_size=2,strides=2)(conv1)
  conv2 = LeakyReLU(0.2)(conv2)

  conv3 = Conv2D(256,kernel_size=2,strides=2)(conv2)
  conv3 = LeakyReLU(0.2)(conv3)

  conv4 = Conv2D(512,kernel_size=2,strides=2)(conv3)
  conv4 = LeakyReLU(0.2)(conv4)

  conv5 = Conv2D(512,kernel_size=2,strides=2)(conv4)
  conv5 = LeakyReLU(0.2)(conv5)

  conv_up_1 = Conv2DTranspose(512,kernel_size=2,strides=2)(conv5)
  conv_up_1 = LeakyReLU(0.2)(conv_up_1)
  conv_up_1 = BatchNormalization()(conv_up_1)
  conv_up_1 = Dropout(0.5)(conv_up_1)

  concat1 = concatenate([conv_up_1,conv4],axis=3)
  conv_up_2 = Conv2DTranspose(256,kernel_size=2,strides=2)(concat1)
  conv_up_2 = LeakyReLU(0.2)(conv_up_2)
  conv_up_2 = BatchNormalization()(conv_up_2)
  conv_up_2 = Dropout(0.5)(conv_up_2)

  concat2 = concatenate([conv_up_2,conv3],axis=3)
  conv_up_3 = Conv2DTranspose(128,kernel_size=2,strides=2)(concat2)
  conv_up_3 = LeakyReLU(0.2)(conv_up_3)
  conv_up_3 = BatchNormalization()(conv_up_3)
  conv_up_3 = Dropout(0.5)(conv_up_3)

  concat3 = concatenate([conv_up_3,conv2],axis=3)
  conv_up_4 = Conv2DTranspose(64,kernel_size=2,strides=2)(concat3)
  conv_up_4 = LeakyReLU(0.2)(conv_up_4)
  conv_up_4 = BatchNormalization()(conv_up_4)
  conv_up_4 = Dropout(0.5)(conv_up_4)

  concat4 = concatenate([conv_up_4,conv1],axis=3)

  output = Conv2D(2,kernel_size=1,strides=1,activation='tanh')(concat4)

  model = Model(
      inputs = input,
      outputs = output
  )

  return model

def discriminator():
  X = Input(shape=in_shape)
  Y = Input(shape=out_shape)
  
  input = Concatenate(axis=3)([X,Y])
  conv1 = Conv2D(64,kernel_size=2,strides=2)(input)
  conv1 = BatchNormalization()(conv1)
  conv1 = LeakyReLU(0.2)(conv1)

  conv2 = Conv2D(128,kernel_size=2,strides=2)(conv1)
  conv2 = BatchNormalization()(conv2)
  conv2 = LeakyReLU(0.2)(co
                         nv2)

  conv3 = Conv2D(256,kernel_size=2,strides=2)(conv2)
  conv3 = BatchNormalization()(conv3)
  conv3 = LeakyReLU(0.2)(conv3)

  conv4 = Conv2D(512,kernel_size=2,strides=2)(conv3)
  conv4 = BatchNormalization()(conv4)
  conv4 = LeakyReLU(0.2)(conv4)

  dense = Flatten()(conv4)
  dense = Dense(128)(dense)
  output = Dense(1,activation='sigmoid')(dense)

  model = Model(
      inputs = [X,Y],
      outputs = output
  )

  return model