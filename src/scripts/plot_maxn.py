import pickle
import os.path
from argparse import ArgumentParser

import numpy as np
from scipy.signal import savgol_filter
import pylab

# MAXN_FILE = '/home/marrabld/data/open_fish_classifier/maxn/annotations.pickle'
# MAXN_FILE = '/home/marrabld/data/ozfish/videos/masn_sebae/A000053_L_60s_slice.pickle'
# MAXN_FILE = '/home/marrabld/data/ozfish/videos/masn_sebae/A000023_L_60s_slice.pkl'
THRESHOLD = 0.95
AIMS_MAXN = 2


def main(args):
    file = open(args.pickle_file, 'rb')
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
            if fish[1] >= args.probability / 100:
                fish_count[fish[0]][ii] += 1
    for item in fish_labels:
        pylab.plot(maxn[0], fish_count[item], '--', label=f'MaxN {item}')
    pylab.xlabel('Frame #')
    pylab.ylabel('Count')
    pylab.legend()
    pylab.grid()

    filename, ext = os.path.splitext(os.path.basename(args.pickle_file))
    output_image_path = os.path.join(os.path.dirname(args.pickle_file), filename + '.maxn.png')

    pylab.savefig(output_image_path)
    if args.show:
        pylab.show()



if __name__ == '__main__':
    parser = ArgumentParser('plot_maxn', 'Given a pick file generated with gen_detections.py')
    parser.add_argument('-f', '--pickle-file', required=True, help='Path to the pickle file with detections')
    parser.add_argument('-p', '--probability', required=False, type=int, default=90)
    parser.add_argument('-s', '--show', required=False, type=bool, default=False)

    args = parser.parse_args()
    exit(main(args) or 0)
