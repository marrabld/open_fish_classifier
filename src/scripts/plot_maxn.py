import pickle
import numpy as np
from scipy.signal import savgol_filter
import pylab

# MAXN_FILE = '/home/marrabld/data/open_fish_classifier/maxn/annotations.pickle'
MAXN_FILE = '/home/marrabld/data/ozfish/videos/masn_sebae/A000053_L_60s_slice.pickle'
THRESHOLD = 0.95
AIMS_MAXN = 2

file = open(MAXN_FILE, 'rb')
maxn = pickle.load(file)

total_count = []
seb_list = []
punct_list = []
frame_list = []
frame_num = 0
label_dict = {}
fish_count = {}
fish_labels = []

for item in maxn[1]:
    for label in item:
       fish_labels.append(label[0])

fish_labels = set(fish_labels)
for label in fish_labels:
    fish_count[label] = np.zeros(len(maxn[0]))

for ii, item in enumerate(maxn[1]):
    frame_num = maxn[0][ii]

    total_count.append(len(item))
    for fish in item:
        if fish[1] >= THRESHOLD:
            fish_count[fish[0]][ii] += 1

for item in fish_labels:
    pylab.plot(maxn[0], fish_count[item], '--', label=f'MaxN {item}')

pylab.xlabel('Frame #')
pylab.ylabel('Count')
pylab.legend()
pylab.grid()
pylab.show()

print('done')
