import os
import shutil

from argparse import ArgumentParser, ArgumentTypeError
from random import shuffle

##
# ArgumentParser 'type' for handling slash-delimited weight triples
##
def weights(raw):
    parts = raw.split('/')

    if len(parts) != 3:
        raise ArgumentTypeError('weights must specify 3 integers i.e 80/10/10')

    values = [ int(p) for p in parts ]

    if sum(values) != 100:
        raise ArgumentTypeError('weights must sum to 100')

    return values

def copy_images(output, type, label, images):
    directory = os.path.join(output, label, type)
    os.makedirs(directory, exist_ok=True)

    for image in images:
        shutil.copyfile(image, os.path.join(directory, os.path.basename(image)))


def main(args):
    if not os.path.exists(args.input):
        print('error: input directory does not exist')
        return 1

    if not os.path.exists(args.metadata):
        print('error: metadata file does not exist')
        return 1

    # first, use the crop metadata to find what crops exist, and divide them into species
    partitions = {}

    with open(args.metadata, 'r') as metadata:
        metadata.readline() # skip header row TODO: validate expected column order?

        for line in metadata:
            # TODO: Could use a proper CSV parser here, but at the moment the file is not complex
            parts = line.rstrip().split(',')
            image = os.path.join(args.input, parts[1])
            partition = '_'.join([ (p or 'unknown').lower() for p in parts[2:5] ])

            if not os.path.exists(image):
                print('warning: unable to find "%s" in input directory' % parts[1])
            else:
                # ensure the partition exists, then add the image to it
                if partition not in partitions:
                    partitions[partition] = []

                partitions[partition].append(image)


    # once partitions have been created, divide each into test/validation/train
    # according to the specified weights
    for label, images in partitions.items():
        shuffle(images)
        total = len(images)

        # train should always be the largest, so compute test/validate first and give the rest to train
        ntest = max(1, round(total * (args.weights[2] / 100)))
        nvalidate = max(1, round(total * (args.weights[1] / 100)))

        copy_images(args.output, 'test', label, images[:ntest])
        copy_images(args.output, 'validate', label, images[ntest:ntest+nvalidate])
        copy_images(args.output, 'train', label, images[ntest+nvalidate:])

    return 0

if __name__ == '__main__':
    parser = ArgumentParser('gen_crop_dataset', description='Utility script for partitioning crops into train/validate/test structures according to family/genus/species')
    parser.add_argument('--metadata', help='path to metadata file in CSV format', required=True)
    parser.add_argument('--input', help='input directory containing crop images', required=True)
    parser.add_argument('--output', help='root directory to output the partitioned crops to', required=True)
    parser.add_argument('--weights', type=weights, help='slash-delimited set of percentage weights to divide the crops in for train/validate/test', required=True)

    args = parser.parse_args()
    exit(main(args) or 0)