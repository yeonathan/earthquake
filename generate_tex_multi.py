import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool

# testset length each: 150_000
# total number of rows: 629_145_480
nrows = (629_145_480 // 150_000) * 150_000
train = pd.read_csv('train.csv', nrows=nrows, \
  iterator=True, chunksize=150_000, \
  dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

cores = 8
core_count = 0
time_step = 121
steps = 149_999 // time_step
a = []

index = 0
counter = 0

def create_hist (alist):
  chunk = alist[0]
  index = alist[1]
  counter = alist[2]
  ac_data = chunk['acoustic_data'] * 1000
  # f, t, Sxx = signal.spectrogram(ac_data, 44000)
  # fig = plt.pcolormesh(t, f, Sxx, cmap=plt.cm.gray)
  # fig.axes.get_xaxis().set_visible(False)
  # fig.axes.get_yaxis().set_visible(False)
  # plt.savefig('/home/bernhard/Documents/ml/earthquake/tex/' + \
  #   str(counter) + '.png', bbox_inches='tight', pad_inches=0)
  ttf = []
  for i in range(time_step):
    ttf.append(chunk['time_to_failure'][index + i * steps])
  np.save('/home/bernhard/Documents/ml/earthquake/tex_ttf/' + str(counter), ttf)
  print('Saved spectrogram ' + str(counter))
  return

for chunk in train:
  if core_count < cores:
    a.append([chunk, index, counter])
    counter += 1
    index += 150_000
    core_count += 1
  if core_count == cores:
    pool = Pool(processes=8)
    _ = pool.map(create_hist, a)
    core_count = 0
    a = []