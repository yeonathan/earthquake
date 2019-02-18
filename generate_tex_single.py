import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# testset length each: 150_000
# total number of rows: 629_145_480
nrows = (629_145_480 // 150_000) * 150_000
# nrows = 150_000
train = pd.read_csv('train.csv', nrows=nrows, \
  iterator=True, chunksize=150_000, \
  dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
#print('train shape', train.shape)

index = 0
counter = 0
steps = 149_999 // 80
X = []
Y = []

for chunk in train:
  x = chunk['acoustic_data'] * 1000
  f, t, Sxx = signal.spectrogram(x, 44000)
  fig = plt.pcolormesh(t, f, Sxx, cmap=plt.cm.gray)
  img = fig.get_array().reshape((334, 256))
  img = np.asarray(img, dtype=int)
  ttf_array = []
  # need 80 timesteps
  for i in range(80):
    ttf_array.append(chunk['time_to_failure'][index + i * steps])
  X.append(img)
  Y.append(ttf_array)
  print('Processed spectrogram ' + str(counter))
  counter += 1
  index += 150_000

np.save('earthquake_X', X)
np.save('earthquake_Y', Y)

