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

# for chunk in train:
#   fig, ax1 = plt.subplots()
#   plt.plot(chunk['acoustic_data'])
#   ax2 = ax1.twinx()
#   plt.plot(chunk['time_to_failure'], color='g')
#   plt.grid(False)
#   plt.show()

# fig, ax1 = plt.subplots()
# plt.plot(train['acoustic_data'])
# ax2 = ax1.twinx()
# plt.plot(train['time_to_failure'], color='g')
# plt.grid(False)
# plt.show()

index = 0
counter = 0
steps = 149_999 // 80
tex = []

for chunk in train:
  x = chunk['acoustic_data'] * 1000
  f, t, Sxx = signal.spectrogram(x, 44000)
  fig = plt.pcolormesh(t, f, Sxx, cmap=plt.cm.gray)
  img = fig.get_array().reshape((334, 256))
  img = np.asarray(img, dtype=int)
  ttf_array = []
  for i in range(80):
    ttf_array.append(chunk['time_to_failure'][index + i * steps])
  a = [img, ttf_array]
  tex.append(a)
  # need 80 timesteps
  # np.save('img_test', img)
  # plt.ylabel('Frequency [Hz]')
  # plt.xlabel('Time [sec]')
  # plt.show()
  # fig.axes.get_xaxis().set_visible(False)
  # fig.axes.get_yaxis().set_visible(False)
  # ttf = chunk['time_to_failure'][index + 149_999]
  # plt.savefig('/home/bernhard/Documents/ml/earthquake/' + \
  #   str(ttf) + '.png', bbox_inches='tight', pad_inches=0)
  print('Processed spectrogram ' + str(counter))
  counter += 1
  index += 150_000

np.save('earthquake_tex', tex)

