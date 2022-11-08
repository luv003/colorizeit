import os
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import *
from helpers import *
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam

plt.style.use('seaborn')

#-- constants
TRAIN_PATH = './datasets/train'
images = np.array(os.listdir(TRAIN_PATH))

opt = Adam(learning_rate=1e-4)
EPOCHS = 50
BATCH_SIZE = 100
HALF_BATCH = 50
N_EXAMPLES = 
N_BATCHES = int(N_EXAMPLES/BATCH_SIZE)
DIS_UPD = 2
GEN_UPD = 1
ones = np.ones((BATCH_SIZE,1))*0.9
zeros = np.zeros((BATCH_SIZE,1))
latest = 0
RERUN = False

#-- Models
gen = generator()
dis = discriminator()

X = Input(shape=in_shape)
Y = Input(shape=out_shape)

pred = gen(X)
comb = dis([X,pred])

pred = Flatten()(pred)
org = Flatten()(Y)

dot = Dot(axes=1,normalize=True)([pred,org])

model = Model(
    inputs = [X,Y],
    outputs = [comb,dot]
)

model.summary()

#-- Training begins here

if RERUN == True:
  gen.load_weights(f'./model_weights/generator{latest}.h5')
  dis.load_weights(f'./model_weights/discriminator{latest}.h5')

for epoch in range(EPOCHS):
  print(f'Epoch {epoch+latest+1}/{EPOCHS}')
  g_loss = 0.0
  d_loss = 0.0

  begin = time.time()
  # print('Training Discriminator:')
  i = list(range(N_EXAMPLES))
  np.random.shuffle(i)
  
  dis.trainable = True
  dis.compile(loss='binary_crossentropy',optimizer=opt)

  for j in range(DIS_UPD):
    for b in range(N_BATCHES):
#       x = grayImages[i[b*BATCH_SIZE:(b+1)*BATCH_SIZE]]
#       y = rgbImages[i[b*BATCH_SIZE:(b+1)*BATCH_SIZE]]
      x,y = next(sample_generator(images,BATCH_SIZE))
      y = (y*2)-1
      if(x.shape[0]==BATCH_SIZE):
          pred = gen.predict(x)
          d_loss+=dis.train_on_batch([x,pred],zeros)
          d_loss+=dis.train_on_batch([x,y],ones)
  dis.trainable = False
  model.compile(loss='binary_crossentropy',optimizer=opt)
  dis.compile(loss='binary_crossentropy',optimizer=opt)
  
  for j in range(GEN_UPD):
    for b in range(N_BATCHES):
#         x = grayImages[i[b*BATCH_SIZE:(b+1)*BATCH_SIZE]]
#         y = rgbImages[i[b*BATCH_SIZE:(b+1)*BATCH_SIZE]]

        x,y = next(sample_generator(images,BATCH_SIZE))
        y = (y*2)-1
        if(x.shape[0]==BATCH_SIZE):
            gl,_,_ = model.train_on_batch([x,y],[ones,ones])
            g_loss += gl
        
  d_loss = d_loss/(DIS_UPD*N_BATCHES)
  g_loss = g_loss/(GEN_UPD*N_BATCHES)

  end = time.time()

  print(f'Generator Loss: {g_loss}\nDiscriminator Loss: {d_loss}')
  print(f'Time Taken: {end-begin} sec\n')

  gloss.append(g_loss)
  dloss.append(d_loss)    

  if (epoch+latest+1)%5==0:
    gen.save_weights(f'./model_weights/generator{epoch+latest+1}.h5')
    dis.save_weights(f'./model_weights/discriminator{epoch+latest+1}.h5')

#-- plots of the generator and discriminator losses over the training
plt.plot(g_loss, c='r', label="Generator Loss")
plt.plot(d_loss, c='b', label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()