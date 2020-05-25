import os
import json
import re
import sys
import shutil
import imagesize

from argparse import ArgumentParser, ArgumentTypeError
from collections import Counter
from random import shuffle
from errno import ENOENT

CATCH_ALL_SPECIES = 'fish'

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

class FriendlyError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

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

def create_dataset_env(name, species, force):
    root_dir = os.path.abspath(os.path.join('datasets', name))

    if os.path.exists(root_dir):
        if not force:
            raise FriendlyError('a "%s" training env already exists, choose a unique name or specify the "force" option' % name)

        # remove the old training data
        log('warning', 'overwriting previous "%s" training env' % name)
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)

    # output the species in this dataset to a file for later retrievel
    with open(os.path.join(root_dir, 'species.list'), 'w') as sf:
        sf.write('\n'.join(species))

    return root_dir

def get_image_size(path):
    try:
        return imagesize.get(path)
    except (OSError, IOError) as e:
        if getattr(e, 'errno', 0) != ENOENT:
            raise
        return None

def extract_frame_data(frame_dir, metadata_path, target_species, catch_all_species):
    metadata = None

    with open(metadata_path, 'r') as mf:
        content = json.load(mf)
        metadata = content['_via_img_metadata']

    for meta in metadata.values():
        frame_path = os.path.join(frame_dir, meta['filename'])
        size = get_image_size(frame_path)

        if size is None:
            log('warning', 'unable to open file "%s"' % frame_path)
            continue

        objects = []

        for region in meta['regions']:
            # Skip any regions that have no label in their attributes
            # TODO: Why are these even in the dataset?
            if 'label' not in region['region_attributes']:
                continue

            shape = region['shape_attributes']
            points = [ shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height'] ]

            # the species labelling in the metadata won't match precisely with the input
            # target species labels so try and map them together here
            search = re.sub(r'\s+', '_', region['region_attributes']['label']).lower()
            species = next((s for s in target_species if s.lower().endswith(search)), catch_all_species)

            if species is not None:
                objects.append((species, points))
        
        yield (frame_path, size, objects)

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
        os.symlink(frame_path, os.path.join(images_dir, name))

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
        target_species = [ s.lower() for s in args.species ]
        catch_all_species = CATCH_ALL_SPECIES if args.enable_catch_all else None

        if catch_all_species:
            if catch_all_species in target_species:
                raise FriendlyError('catch-all species (%s) is included in target species list')

            target_species.append(CATCH_ALL_SPECIES)

         # Set up the training environment and extract the dataset
        root_dir = create_dataset_env(args.name, target_species, args.force_overwrite)

        log('info', 'generating training dataset')

        frames = list(extract_frame_data(args.frame_directory, args.metadata_path, target_species, catch_all_species))
        
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
    except FriendlyError as err:
        log('error', err)
        return 1
    
    return 0
   
if __name__ == '__main__':
    parser = ArgumentParser('gen_frame_dataset', 'Use an AIMS-provided metadata file to generate an ImageAI training dataset')
    parser.add_argument('-w', '--weights', required=True, type=weights, help='Slash-delimited set of percentage weights to divide the crops in for train/validate/test')
    parser.add_argument('-d', '--frame-directory', required=True, help='Root directory containing all of the frame images')
    parser.add_argument('-m', '--metadata-path', required=True, help='Path to the metadata file describing bounding boxes within frames')
    parser.add_argument('-s', '--species', required=True, action='append', help='Species to train the model on, in "genus_family_species" format (can be specified multiple times)')
    parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous dataset with the same name', default=False)
    parser.add_argument('--enable-catch-all', required=False, action='store_true', help='Toggle if a catch all "fish" label should be used for species not in the species list. Without this flag, annotations for species not in the target species are omitted')
    parser.add_argument('name', help='Name of the dataset, must be unique unless "--force-overwrite" was specified')

    args = parser.parse_args()
    exit(main(args) or 0)
