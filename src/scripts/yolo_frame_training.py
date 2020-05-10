import os
import cv2
import json
import re
import sys
import shutil

from argparse import ArgumentParser, ArgumentTypeError
from collections import Counter
from random import shuffle

ANNOTATION_FORMAT = """
<annotation>
    <folder>{folder}</folder>
    <filename>{filename}</filename>
    <path>{path}</path>
    <source><database>Unknown</database></source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    {object_xml}

</annotation>
"""

ANNOTATION_OBJECT_FORMAT = """
<object>
    <name>{label}</name>
    <pose>Unspecified</pose>
    <bndbox>
        <xmin>{xmin}</xmin>
        <ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax>
        <ymax>{ymax}</ymax>
    </bndbox>
</object>
"""

def weights(raw):
    parts = raw.split('/')

    if len(parts) != 3:
        raise ArgumentTypeError('must specify 3 integer weights, i.e 80/10/10')

    values = [ int(p) for p in parts ]

    if sum(values) != 100:
        raise ArgumentTypeError('weights must sum to 100')

    return values

def log(level, message):
    output = sys.stderr if level == 'error' else sys.stdout
    output.write('%s: %s\n' % (level, message))

def gen_annotation(path, size, objects):
    object_xml = ''

    for label, points in objects:
        object_xml += ANNOTATION_OBJECT_FORMAT.format(
            label=label, 
            xmin=points[0], 
            ymin=points[1],
            xmax=points[2],
            ymax=points[3]
        )

    return ANNOTATION_FORMAT.format(
        folder = os.path.dirname(path),
        filename = os.path.basename(path),
        path = path,
        width = size[0],
        height = size[1],
        object_xml = object_xml
    )

def shape_to_points(shape):
    return 

def create_training_env(name, force):
    root_dir = os.path.abspath(os.path.join('training', name))

    if os.path.exists(root_dir):
        if not force:
            raise ValueError('a "%s" training env already exists, choose a unique name or specify the "force" option' % name)

        # remove the old training data
        log('warning', 'overwriting previous "%s" training env' % name)
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)
    return root_dir

def map_species_label(label, target_species):
    search = re.sub(r'\s+', '_', label).lower()
    return next((species for species in target_species if species.lower().endswith(search)), None)

def extract_objects(regions, target_species):
    for region in regions:
        if 'label' in region['region_attributes']:
            label = map_species_label(region['region_attributes']['label'], target_species)

            if label is not None:
                shape = region['shape_attributes']
                points = [ shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height'] ]
                yield (label, points)

def extract_frame_data(frame_dir, metadata_path, species):
    metadata = None

    with open(metadata_path, 'r') as mf:
        content = json.load(mf)
        metadata = content['_via_img_metadata']

    for _, meta in metadata.items():
        objects = list(extract_objects(meta['regions'], species))

        if len(objects) > 0:
            frame_path = os.path.join(frame_dir, meta['filename'])
            img = cv2.imread(frame_path)

            if img is None:
                log('warning', 'unable to open file "%s"' % frame_path)
            else:
                yield (frame_path, (img.shape[1], img.shape[0]), objects)

def partition_frames(frames, weights):
    shuffle(frames)
    total = len(frames)

    # train should always be the largest, so compute test/validate first and give the rest to train
    ntest = max(1, round(total * (weights[2] / 100)))
    nvalidation = max(1, round(total * (weights[1] / 100)))

    return (frames[ntest+nvalidation:], frames[ntest:ntest+nvalidation], frames[:ntest])

def create_training_dataset(root_dir, name, dataset):
    images_dir = os.path.join(root_dir, name, 'images')
    annotations_dir = os.path.join(root_dir, name, 'annotations')

    os.makedirs(images_dir)
    os.makedirs(annotations_dir)

    for frame_path, size, objects in dataset:
        name = os.path.basename(frame_path)
        os.link(frame_path, os.path.join(images_dir, name))

        with open(os.path.join(annotations_dir, os.path.splitext(name)[0] + '.xml'), 'w') as af:
            af.write(gen_annotation(frame_path, size, objects))

def log_dataset_summary(name, frames):
    label_counts = Counter((label for _, _, objects in frames for label,_ in objects))
    total_objects = sum((len(objects) for _, _, objects in frames))
    log('info', '%d frames, with %d objects in "%s" dataset' % (len(frames), total_objects, name))
    
    for label, total in sorted(label_counts.items(), reverse=True, key=lambda x: x[1]):
        log('info', '    %s: %d' % (label, total))
    

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
    # Sanity check inputs
    if not os.path.isdir(args.frame_directory):
        log('error', 'unable to find directory "%s"' % args.frame_directory)
        return 1

    if not os.path.isfile(args.metadata_path):
        log('error', 'unable to find file "%s"' % args.metadata_path)
        return 1

    try:
         # Set up the training environment and extract the dataset
        root_dir = create_training_env(args.name, args.force_overwrite)

        log('info', 'generating training dataset')

        frames = list(extract_frame_data(args.frame_directory, args.metadata_path, args.species))
        
        if len(frames) < 3:
            log('error', 'not enough frame data; minimum 3 frames required, got %d' % len(frames))
            return 1

        # partition the dataset into train/validation/test according to weights
        train, validation, test = partition_frames(frames, args.weights)

        create_training_dataset(root_dir, 'train', train)
        create_training_dataset(root_dir, 'validation', validation)

        # This is a bit hacky, but set up another 'train' directory in the test folder 
        # This is to fool ImageAI's "evaluateModel" being opinionated on the directory structure.
        create_training_dataset(os.path.join(root_dir, 'test'), 'train', test)

        log_dataset_summary('train', train)
        log_dataset_summary('validation', validation)
        log_dataset_summary('test', test)

        log('info', 'training model')
        train_model(root_dir, args.species, args.epochs, args.batch_size, args.pretrained_path)
    except ValueError as err:
        log('error', err)
        return 1
    
    return 0
   
if __name__ == '__main__':
    parser = ArgumentParser('yolo_frame_training', 'Use ImageAI to run YOLOv3 train on fish frame data')
    parser.add_argument('-e', '--epochs', required=False, type=int, help='Number of epochs to run the training for', default=100)
    parser.add_argument('-w', '--weights', required=True, type=weights, help='Slash-delimited set of percentage weights to divide the crops in for train/validate/test')
    parser.add_argument('-d', '--frame-directory', required=True, help='Root directory containing all of the frame images')
    parser.add_argument('-m', '--metadata-path', required=True, help='Path to the metadata file describing bounding boxes within frames')
    parser.add_argument('-s', '--species', required=True, action='append', help='Species to train the model on, in "genus_family_species" format (can be specified multiple times)')
    parser.add_argument('-p', '--pretrained-path', required=True, help='Path to pre-trained YOLO model to apply transfer learning from')
    parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous training run with the same name', default=False)
    parser.add_argument('-b', '--batch-size', required=False, type=int, help='Batch size for model training (default: 4)', default=4)
    parser.add_argument('name', help='Name of the training run, must be unique unless "--force-overwrite" was specified')

    args = parser.parse_args()
    exit(main(args) or 0)
