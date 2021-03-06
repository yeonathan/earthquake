from PIL import Image
import os
import numpy as np
from itertools import islice

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

num_tex = 1000

def load_images(path):
  image_list = os.listdir(path)
  loaded_images = []
  for image in islice(image_list, num_tex):
    with open(os.path.join(path, image), 'rb') as i:
      img = Image.open(i)
      data = np.asarray(img, dtype='int32')
      data = data[:,:,0]
      loaded_images.append(data)
  loaded_images = np.array(loaded_images)
  return loaded_images

def load_ttf(path):
  ttf_list = os.listdir(path)
  loaded_ttf = []
  for ttf in islice(ttf_list, num_tex):
    with open(os.path.join(path, ttf), 'rb') as i:
      data = np.load(i)
      loaded_ttf.append(data)
  loaded_ttf = np.array(loaded_ttf)
  return loaded_ttf

def model(input_shape):

  X_input = Input(shape = input_shape)
  
  X = Conv1D(50, kernel_size=15, strides=4)(X_input)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Dropout(0.8)(X)

  X = GRU(units = 128, return_sequences = True)(X)
  X = Dropout(0.8)(X)
  X = BatchNormalization()(X)
  
  
  X = GRU(units = 128, return_sequences = True)(X)
  X = Dropout(0.8)(X)
  X = BatchNormalization()(X)
  X = Dropout(0.8)(X)
  
  X = TimeDistributed(Dense(1, activation = "relu"))(X)

  model = Model(inputs = X_input, outputs = X)
  
  return model

images = load_images('/home/bernhard/Documents/ml/earthquake/tex_images/')
ttf = load_ttf('/home/bernhard/Documents/ml/earthquake/tex_ttf/')
ttf = ttf.reshape((num_tex, 121, 1))

# print(images.shape)
# print(ttf.shape)

model = model(input_shape = (496, 369))
# model.summary()

# model = load_model('./models/tr_model.h5')

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

# model.fit(images, ttf, batch_size = 5, epochs=10)
# model.save_weights('eq_weights_squared.h5')

model.load_weights('eq_weights_crossentropy.h5')
img = images[500].reshape((1, 496, 369))
preds = model.predict(img, verbose=0)[0]
preds = np.asarray(preds).astype('float64')

for i in preds:
  print(i)

print('')

for i in ttf[100]:
  print(i)
