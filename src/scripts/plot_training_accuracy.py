import pylab
import numpy as np

# TRAIN_FILE = "/home/marrabld/data/yolo/model_jason_12/eval-top-two-70-15-15-b4.out"
TRAIN_FILE = "/home/marrabld/data/yolo/model_eval_home/0.9-eval-home-rerun.out"
# TRAIN_FILE = "/home/marrabld/data/yolo/model_eval_12fish/training_results.txt"
# TRAIN_FILE = "/home/marrabld/data/yolo/model_eval_top_twelve/eval-top-twelve-training.out"
dict_list = []

with open(TRAIN_FILE) as f:
    train = {}
    for line in f:
        if not ":" in line and not "=" in line:
            pass
        elif ":" in line:
            tmp = line.split(":")
            # deal with edge cases
            train[tmp[0]] = tmp[1]  # split into key value pairs.
            if ".h5" in tmp[1]:
                train['epoch'] = np.int(tmp[1].split("-ex-")[1].split("--loss")[0].lstrip("0"))
        else:
            dict_list.append(train)
            train = {}

keys = list(dict_list[0].keys())
rm_words = ['info', 'Model File', 'epoch', 'Using IoU ', 'Using Object Threshold ', 'Using Non-Maximum Suppression ', 'mAP']
for item in rm_words:
    keys.remove(item)

map = []
e = []


from collections import defaultdict
plot_dict = defaultdict(list)


for item in dict_list:
    map.append(item['mAP'])
    e.append(item["epoch"])
    for ii, fish in enumerate(keys):
        plot_dict[fish].append(item[fish])


for item in keys:
    pylab.plot(np.asarray(e, dtype=np.int), np.asarray(plot_dict[item], dtype=float), 'o--', alpha=0.6, label=item)
    # pylab.semilogx(np.asarray(e, dtype=np.int), np.asarray(plot_dict[item], dtype=float), 'o--', alpha=0.6, label=item)
    # pylab.yticks(np.arange(0, 1, step=0.1))

pylab.plot(np.asarray(e, dtype=np.int), np.asarray(map, dtype=np.float), 'o--', alpha=0.6, label='mAP')
pylab.grid()


pylab.ylabel('Mean Accuracy')
pylab.xlabel('Epoch')
# pylab.xlim([0, 150])
pylab.legend()
pylab.show()
