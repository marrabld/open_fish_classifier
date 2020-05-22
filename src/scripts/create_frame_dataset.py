import shutil
import os
import sys
import re
import json
import imagesize

from argparse import ArgumentParser, ArgumentTypeError
from collections import Counter, OrderedDict
from random import shuffle
from errno import ENOENT

CATCH_ALL_SPECIES = 'fish'

VOC_ANNOTATION_FORMAT = """
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

VOC_ANNOTATION_OBJECT_FORMAT = """
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

class DatasetSpecification:
    def __init__(self, train, validate, test):
        self.train = train  
        self.validate = validate
        self.test = test

    def summarize(self):
        total_frames = 
        return OrderedDict([(key, self._summarize_partition(getattr(self, key))) for key in ['test','validate','train']])

    def _summarize_partition(self, partition):  
        label_totals = Counter((label for _, objects in partition for label, *_ in objects))

        totals = OrderedDict()
        totals['total_frames'] = len(partition)
        totals['total_objects'] = sum((len(objects) for _, objects in partition))

        for label, total in sorted(label_totals.items(), reverse=True, key=lambda x: x[1]):
            totals[label] = total

        return totals

# Helper function to parse weights passed on the command-line
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

def get_image_size(path):
    try:
        return imagesize.get(path)
    except (OSError, IOError) as e:
        if getattr(e, 'errno', 0) != ENOENT:
            raise
        return None

def gen_voc_annotation(frame_path, size, objects):
    object_xml = ''

    for label, x1, y1, x2, y2 in objects:
        object_xml += VOC_ANNOTATION_OBJECT_FORMAT.format(
            label=label, 
            xmin=x1, 
            ymin=y1,
            xmax=x2,
            ymax=y2
        )

    return VOC_ANNOTATION_FORMAT.format(
        folder = os.path.dirname(frame_path),
        filename = os.path.basename(frame_path),
        path = frame_path,
        width = size[0],
        height = size[1],
        object_xml = object_xml
    )

def init_dataset_env(name, force):
    root_dir = os.path.abspath(os.path.join('datasets', name))

    if os.path.exists(root_dir):
        if not force:
            raise FriendlyError('a "%s" training env already exists, choose a unique name or specify the "force" option' % name)

        # remove the old training data
        log('warning', 'overwriting previous "%s" training env' % name)
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)
    return root_dir

def create_dataset_partition(root_dir, frame_dir, name, partition):
    images_dir = os.path.join(root_dir, name, 'images')
    annotations_dir = os.path.join(root_dir, name, 'annotations')

    os.makedirs(images_dir)
    os.makedirs(annotations_dir)

    for frame_name, objects in partition:
        frame_path = os.path.join(frame_dir, frame_name)
        size = get_image_size(frame_path)

        if size is None:
            raise FriendlyError('unable to find frame "%s"' % frame_name)

        # symlink the original image into the dataset
        # TODO: optional flag for 'copy' behaviour?
        os.symlink(frame_path, os.path.join(images_dir, frame_name))

        # write the VOC annotatioons file
        # TODO: Also output a darknet file?
        with open(os.path.join(annotations_dir, os.path.splitext(frame_name)[0] + '.xml'), 'w') as af:
            af.write(gen_voc_annotation(frame_path, size, objects))

def log_summary(summary):
    for partition_key in summary:
        p_summary = summary[partition_key]
        log('info', '"%s" partition contains %d objects from %d frames' % (partition_key, p_summary['total_objects'], p_summary['total_frames']))

        for key in p_summary:
            if key not in ('total_objects','total_frames'):
                log('info', '    %s: %d' % (key, p_summary[key]))

def create_dataset(root_dir, frame_dir, spec):
    create_dataset_partition(root_dir, frame_dir, 'train', spec.train)
    create_dataset_partition(root_dir, frame_dir, 'validation', spec.validate)
    # Test directory structure is a little weird to get around ImageAI's hardcoded rules
    # with where it looks for images during evaluateModel()
    create_dataset_partition(os.path.join(root_dir, 'test'), frame_dir, 'train', spec.test)

def extract_frames(metadata_path, target_species, catch_all_species):
    metadata = None
    frames = []

    with open(metadata_path, 'r') as mf:
        content = json.load(mf)
        metadata = content['_via_img_metadata']

    for meta in metadata.values():
        objects = []

        for region in meta['regions']:
            # Skip any regions that have no label in their attributes
            # TODO: Why are these even in the dataset?
            if 'label' not in region['region_attributes']:
                continue

            shape = region['shape_attributes']
            x1, y1, x2, y2 = shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height']
            
            # the species labelling in the metadata won't match precisely with the input
            # target species labels so try and map them together here
            search = re.sub(r'\s+', '_', region['region_attributes']['label']).lower()
            species = next((s for s in target_species if s.lower().endswith(search)), catch_all_species)

            if species is not None:
                objects.append((species, x1, y1, x2, y2))

        # each metadata entry corresponds to a single frame
        # it doesn't matter if the frame contains no annotated objects
        frames.append((meta['filename'], objects))

    return frames

def partition_frames(frames, weights):
    shuffle(frames)
    total = len(frames)

    # train should always be the largest, so compute test/validate first and give the rest to train
    ntest = max(1, round(total * (weights[2] / 100)))
    nvalidation = max(1, round(total * (weights[1] / 100)))

    return (frames[ntest+nvalidation:], frames[ntest:ntest+nvalidation], frames[:ntest])

def generate(args):
    if not os.path.isfile(args.metadata_path):
        raise FriendlyError('unable to find file "%s"' % args.metadata_path)

    target_species = [ s.lower() for s in args.species ]
    catch_all_species = CATCH_ALL_SPECIES if args.enable_catch_all else None

    if catch_all_species:
        if catch_all_species in target_species:
            raise FriendlyError('catch-all species (%s) is included in target species list')
        target_species.append(CATCH_ALL_SPECIES)

    if len(target_species) == 0:
        raise FriendlyError('at least one target species, or --enable-catch-all must be specified')

    # collect all the frames in the metadata file and partition them according to the requested weighting
    frames = extract_frames(args.metadata_path, target_species, catch_all_species)

    if len(frames) < 3:
        raise FriendlyError('not enough frame data with the provided species list; minimum 3 frames required, got %d' % len(frames))

    train, validate, test = partition_frames(frames, args.weights)
    return DatasetSpecification(train, validate, test)

def clone(args):
    try:
        with open(args.specification_path, 'r') as spec_file:
            raw_spec = json.load(spec_file)
            return DatasetSpecification(raw_spec['train'], raw_spec['validate'], raw_spec['test'])
    except (OSError, IOError) as e:
        if getattr(e, 'errno', 0) != ENOENT:
            raise
        raise FriendlyError('unable to open file "%s"' % args.specification_path)

def main(args):
    # Perform common initialisation work and sanity checking
    try:
        if not os.path.isdir(args.frame_directory):
            raise FriendlyError('unable to find directory "%s"' % args.frame_directory)

        # init the environment and obtain the dataset spec
        root_dir = init_dataset_env(args.name, args.force_overwrite)
        spec = generate(args) if args.command == 'generate' else clone(args)

        create_dataset(root_dir, args.frame_directory, spec)

        # write the specification
        with open(os.path.join(root_dir, 'specification.json'), 'w') as spec_file:
            json.dump(spec.__dict__, spec_file)

        # write the summary
        with open(os.path.join(root_dir, 'summary.json'), 'w') as summary_file:
            summary = spec.summarize()
            json.dump(summary, summary_file, indent=2)
            log_summary(summary)

    except FriendlyError as err:
        log('error', err)
        return 1

if __name__ == '__main__':
    parser = ArgumentParser('create_frame_dataset', description='Creates a model training dataset')

    # Create sub-parsers to allow for either generating a new dataset, or cloning an existing one
    subparsers = parser.add_subparsers(dest='command', required=True, help='Datasets can either be generated from an AIMS metadata file, or be cloned from a previously-created dataset')
    
    generate_parser = subparsers.add_parser('generate', help='Generate a new dataset from an AIMS frame metadata file')
    generate_parser.add_argument('-w', '--weights', required=True, type=weights, help='Slash-delimited set of percentage weights to divide the crops in for train/validate/test')
    generate_parser.add_argument('-m', '--metadata-path', required=True, help='Path to the AIMS metadata file describing bounding boxes within frames')
    generate_parser.add_argument('-s', '--species', required=True, action='append', help='Species to train the model on, in "genus_family_species" format (can be specified multiple times)')
    generate_parser.add_argument('-d', '--frame-directory', required=True, help='Root directory containing all of the frame images')
    generate_parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous dataset with the same name', default=False)
    generate_parser.add_argument('--enable-catch-all', required=False, action='store_true', help='Toggle if a catch all "fish" label should be used for species not in the species list. Without this flag, annotations for species not in the target species are omitted')
    generate_parser.add_argument('name', help='Name of the dataset, must be unique unless "--force-overwrite" was specified')

    clone_parser = subparsers.add_parser('clone', help='Clone an existing dataset using its dataset specification')
    clone_parser.add_argument('-s', '--specification-path', required=True, help='Path to specification file')
    clone_parser.add_argument('-d', '--frame-directory', required=True, help='Root directory containing all of the frame images')
    clone_parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous dataset with the same name', default=False)
    clone_parser.add_argument('name', help='Name of the dataset, must be unique unless "--force-overwrite" was specified')

    args = parser.parse_args()
    print(args.command)
    exit(main(args) or 0)

