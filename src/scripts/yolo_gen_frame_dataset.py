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

DEFAULT_SPECIES = 'fish'

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

def create_dataset_env(name, force):
    root_dir = os.path.abspath(os.path.join('datasets', name))

    if os.path.exists(root_dir):
        if not force:
            raise ValueError('a "%s" training env already exists, choose a unique name or specify the "force" option' % name)

        # remove the old training data
        log('warning', 'overwriting previous "%s" training env' % name)
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)
    return root_dir

def map_species_label(label, target_species, default_species):
    search = re.sub(r'\s+', '_', label).lower()
    return next((species for species in target_species if species.lower().endswith(search)), default_species)

def extract_objects(regions, target_species, default_species):
    for region in regions:
        if 'label' in region['region_attributes']:
            shape = region['shape_attributes']
            points = [ shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height'] ]
            label = map_species_label(region['region_attributes']['label'], target_species, default_species)
            yield (label, points)

def extract_frame_data(frame_dir, metadata_path, species):
    metadata = None

    with open(metadata_path, 'r') as mf:
        content = json.load(mf)
        metadata = content['_via_img_metadata']

    for _, meta in metadata.items():
        objects = list(extract_objects(meta['regions'], species, DEFAULT_SPECIES))

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

def create_training_partition(root_dir, name, partition):
    images_dir = os.path.join(root_dir, name, 'images')
    annotations_dir = os.path.join(root_dir, name, 'annotations')

    os.makedirs(images_dir)
    os.makedirs(annotations_dir)

    for frame_path, size, objects in partition:
        name = os.path.basename(frame_path)
        os.link(frame_path, os.path.join(images_dir, name))

        with open(os.path.join(annotations_dir, os.path.splitext(name)[0] + '.xml'), 'w') as af:
            af.write(gen_annotation(frame_path, size, objects))

def log_partition_summary(name, frames):
    label_counts = Counter((label for _, _, objects in frames for label,_ in objects))
    total_objects = sum((len(objects) for _, _, objects in frames))
    log('info', '%d frames, with %d objects in "%s" partition' % (len(frames), total_objects, name))
    
    for label, total in sorted(label_counts.items(), reverse=True, key=lambda x: x[1]):
        log('info', '    %s: %d' % (label, total))
        
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
        root_dir = create_dataset_env(args.name, args.force_overwrite)

        log('info', 'generating training dataset')

        frames = list(extract_frame_data(args.frame_directory, args.metadata_path, args.species))
        
        if len(frames) < 3:
            log('error', 'not enough frame data; minimum 3 frames required, got %d' % len(frames))
            return 1

        # partition the dataset into train/validation/test according to weights
        train, validation, test = partition_frames(frames, args.weights)

        create_training_partition(root_dir, 'train', train)
        create_training_partition(root_dir, 'validation', validation)

        # This is a bit hacky, but set up another 'train' directory in the test folder 
        # This is to fool ImageAI's "evaluateModel", which is opinionated on the directory structure.
        create_training_partition(os.path.join(root_dir, 'test'), 'train', test)

        log_partition_summary('train', train)
        log_partition_summary('validation', validation)
        log_partition_summary('test', test)
    except ValueError as err:
        log('error', err)
        return 1
    
    return 0
   
if __name__ == '__main__':
    parser = ArgumentParser('yolo_frame_training', 'Use an AIMS-provided metadata file to generate an ImageAI+YOLOv3 training dataset')
    parser.add_argument('-w', '--weights', required=True, type=weights, help='Slash-delimited set of percentage weights to divide the crops in for train/validate/test')
    parser.add_argument('-d', '--frame-directory', required=True, help='Root directory containing all of the frame images')
    parser.add_argument('-m', '--metadata-path', required=True, help='Path to the metadata file describing bounding boxes within frames')
    parser.add_argument('-s', '--species', required=True, action='append', help='Species to train the model on, in "genus_family_species" format (can be specified multiple times)')
    parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous dataset with the same name', default=False)
    parser.add_argument('name', help='Name of the dataset, must be unique unless "--force-overwrite" was specified')

    args = parser.parse_args()
    exit(main(args) or 0)
