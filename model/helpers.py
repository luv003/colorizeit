import os
import math
import cv2
import numpy as np

def sample_generator(images,batch_size=128):
  n_samples = images.shape[0]
  while True:
    np.random.shuffle(images)
    for off in range(0,n_samples,batch_size):
      samples = images[off:off+batch_size]
      light = []
      ab = []
      for path in samples:
        try:
          ext = path[-4:]
          if ext == 'JPEG':
              img = cv2.imread('../input/imagenet-no-dir/tiny-imagenet/'+path)
              img = cv2.resize(img,(64,64))
              img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
              lab_img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
              light.append(lab_img[:,:,0].astype('float32'))
              ab.append(lab_img[:,:,1:].astype('float32'))
        except:
          print(f'Error with file {path}\n')
          continue
      
      light = np.asarray(light).reshape(-1,64,64,1)/255.0
      ab = np.asarray(ab).reshape(-1,64,64,2)/255.0

      yield light,ab

def plotImages(gray,rgb,pred):
  n = gray.shape[0]
  fig = plt.figure(figsize=(10,15))
  fig.suptitle('Grayscale vs Ground Truth vs Prediction')
  for i in range(gray.shape[0]):
    plt.subplot(n,3,3*i+1)
    plt.imshow(gray[i].reshape(64,64),cmap='gray',interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(n,3,3*i+2)
    plt.imshow(rgb[i],interpolation='nearest')
    plt.axis('off')

    plt.subplot(n,3,3*i+3)
    plt.imshow(pred[i],interpolation='nearest')
    plt.axis('off')

  plt.tight_layout()
  plt.show()