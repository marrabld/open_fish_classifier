import pickle
import os.path
from argparse import ArgumentParser

import numpy as np
import pylab

pylab.rcParams["figure.figsize"] = (15, 4)
pylab.rcParams["figure.dpi"] = 150

CROP_START = 10000
CROP_END = 10000


def main(args):
    file = open(args.pickle_file, 'rb')
    maxn = pickle.load(file)
    total_count = []
    fish_count = {}
    fish_labels = []
    for item in maxn[1]:
        for label in item:
            fish_labels.append(label[0])
    fish_labels = set(fish_labels)
    for label in fish_labels:
        fish_count[label] = np.zeros(len(maxn[0]))
    for ii, item in enumerate(maxn[1]):
        total_count.append(len(item))
        for fish in item:
            if fish[1] >= args.probability / 100:
                fish_count[fish[0]][ii] += 1
    for item in fish_labels:
        tran = 0.3 if item == 'fish' else 0.5
        tran = 1 if item == 'lethrinus_punctulatus' else tran
        punct_c = None if item == 'lethrinus_punctulatus' else None
        z_order = 100 if item != 'lethrinus_punctulatus' else 10

        pylab.plot(np.asarray(maxn[0][CROP_START:-CROP_END]), fish_count[item][CROP_START:-CROP_END], '-',
                   label=f"{item.split('_')[-1]}", color=punct_c, alpha=tran, zorder=z_order)

    if args.max_n:
        # pylab.hlines(args.max_n, CROP_START, CROP_END, colors='r', linestyles='solid', label='MaxN')
        pylab.plot([maxn[0][CROP_START], maxn[0][-CROP_END]], [args.max_n, args.max_n], '-', color='r',
                   zorder=300)

    if args.frame_number:
        # fn = maxn[0].index(args.frame_number)
        pylab.plot(args.frame_number, args.max_n, 'o', color='r', zorder=300, label='MaxN')

        # pylab.hist(fish_count[item], label=f'MaxN {item}')
    pylab.xlabel(f"{os.path.basename(args.pickle_file).split('_')[0]} Frame #")
    pylab.ylabel('Fish count')
    pylab.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    pylab.tight_layout(rect=[0, 0, 1, 1])
    # pylab.title()
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
    parser.add_argument('-m', '--max-n', required=False, type=int, default=None)
    parser.add_argument('-n', '--frame-number', required=False, type=int, default=None)

    args = parser.parse_args()
    exit(main(args) or 0)
