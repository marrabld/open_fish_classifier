import os
import annotate

from argparse import ArgumentParser

def convert_annotations(base_dir, species_list):
    image_dir = os.path.abspath(os.path.join(base_dir, 'images'))

    with open(os.path.join(base_dir, 'darknet.data'), 'w') as out_file:
        with os.scandir(os.path.join(base_dir, 'annotations')) as scanner:
            for entry in scanner:
                filename, ext = os.path.splitext(entry.name)
                if not entry.is_file() or ext != '.xml':
                    continue

                image = annotate.read_voc(entry.path)
                annotate.write_darknet(image, species_list)
                out_file.write(image.path + '\n')

def main(args):
    root_dir = os.path.join('.', 'datasets', args.dataset)
    species_path = os.path.join(root_dir, 'species.list')

    if not os.path.isdir(root_dir):
        print('error: unable to find a dataset named "%s"' % args.dataset)
        return 1

    # read the species list out so we can convert names -> indices for darknet
    with open(species_path, 'r') as species_file:
        lines = species_file.readlines()
        species_list = [ line.strip() for line in lines if line ]

    # convert all the train/validation annotations to darknet
    convert_annotations(os.path.join(root_dir, 'train'), species_list)
    convert_annotations(os.path.join(root_dir, 'validation'), species_list)

if __name__ == '__main__':
    parser = ArgumentParser('gen_darknet', description='Generate Darknet files for an existing dataset')
    parser.add_argument('dataset', help='Name of the dataset to generate Darknet files for')

    exit(main(parser.parse_args()) or 0)
