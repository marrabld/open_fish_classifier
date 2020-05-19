import pickle
import numpy as np
from scipy.signal import savgol_filter
import pylab

MAXN_FILE = '/home/marrabld/data/open_fish_classifier/maxn/annotations.pickle'
THRESHOLD = 0.9
AIMS_MAXN = 2

file = open(MAXN_FILE, 'rb')
maxn = pickle.load(file)

total_count = []
seb_list = []
punct_list = []
frame_list = []
frame_num = 0

for item in maxn:
    frame_num += 1
    seb_count = 0
    punct_count = 0
    total_count.append(len(item))
    for fish in item:
        if 'lutjanidae_lutjanus_sebae' in fish[0] and fish[1] >= THRESHOLD:
            seb_count += 1
        elif 'lethrinidae_lethrinus_punctulatus' in fish[0] and fish[1] >= THRESHOLD:
            punct_count += 1

    seb_list.append(seb_count)
    frame_list.append(frame_num)
    punct_list.append(punct_count)

pylab.plot(frame_list, total_count, label='Total Fish Detected', alpha=0.25)
#punct_list = savgol_filter(np.asarray(punct_list), 51, 2)
pylab.plot(frame_list, punct_list, '--', label='lethrinidae_lethrinus_punctulatus', alpha=0.6)
#seb_list = savgol_filter(np.asarray(seb_list), 51, 2)
pylab.plot(frame_list, seb_list, '--', label='lutjanidae_lutjanus_sebae', alpha=0.6)

pylab.plot(41838, AIMS_MAXN, '*', c='r')
pylab.plot([0, len(frame_list)], [2, 2], '--', c='r', label=f'MaxN {AIMS_MAXN}')
pylab.ylim([0, 25])
pylab.xlabel('Frame #')
pylab.ylabel('Count')
pylab.legend()
pylab.grid()
pylab.show()

print('done')
