import os
import cv2
import json
import re
import sys
import shutil

from argparse import ArgumentParser

DEFAULT_SPECIES = 'fish'

def log(level, message):
    output = sys.stderr if level == 'error' else sys.stdout
    output.write('%s: %s\n' % (level, message))

def create_training_env(name, dataset_name, force):
    root_dir = os.path.abspath(os.path.join('training', name))
    dataset_dir = os.path.abspath(os.path.join('datasets', dataset_name))

    if not os.path.isdir(dataset_dir):
        raise ValueError('no dataset named "%s" found in datasets directory' % dataset_name)

    if os.path.exists(root_dir):
        if not force:
            raise ValueError('a "%s" training env already exists, choose a unique name or specify the "force" option' % name)

        # remove the old training data
        log('warning', 'overwriting previous "%s" training env' % name)
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)
    os.symlink(os.path.join(dataset_dir, 'train'), os.path.join(root_dir, 'train'), target_is_directory=True)
    os.symlink(os.path.join(dataset_dir, 'validation'), os.path.join(root_dir, 'validation'), target_is_directory=True)
    os.symlink(os.path.join(dataset_dir, 'test'), os.path.join(root_dir, 'test'), target_is_directory=True)

    return root_dir

def train_model(root_dir, species, epochs, batch_size, pretrained_path):
    from imageai.Detection.Custom import DetectionModelTrainer
    import tensorflow as tf

    if not tf.test.is_gpu_available():
        log('warning', 'GPU support is not available, training on CPU')

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=root_dir)
    trainer.setTrainConfig(object_names_array=species, batch_size=batch_size, train_from_pretrained_model=pretrained_path, num_experiments=epochs)

    trainer.trainModel()
        
def main(args):
    root_dir = create_training_env(args.name, args.dataset, args.force_overwrite)

    # Sanity check inputs
    if not os.path.isdir(root_dir):
        log('error', 'unable to find training dataset named "%s"' % args.name)
        return 1

    log('info', 'training model')
    train_model(root_dir, [*args.species, DEFAULT_SPECIES], args.epochs, args.batch_size, args.pretrained_path)

    return 0
   
if __name__ == '__main__':
    parser = ArgumentParser('yolo_frame_training', 'Use ImageAI to run YOLOv3 train on fish frame data')
    parser.add_argument('-e', '--epochs', required=False, type=int, help='Number of epochs to run the training for', default=200)
    parser.add_argument('-p', '--pretrained-path', required=True, help='Path to pre-trained YOLO model to apply transfer learning from', default='')
    parser.add_argument('-b', '--batch-size', required=False, type=int, help='Batch size for model training (default: 4)', default=4)
    parser.add_argument('-s', '--species', required=True, action='append', help='Species to train the model on, in "genus_family_species" format (can be specified multiple times)')
    parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous dataset with the same name', default=False)
    parser.add_argument('-d', '--dataset', required=True, help='Name of dataset to use for training, typically created using yolo_gen_frame_dataset.py')
    parser.add_argument('name', help='Name of the unique training run')

    args = parser.parse_args()
    exit(main(args) or 0)
