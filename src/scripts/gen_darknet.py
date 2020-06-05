import os
import xml.etree.ElementTree as ET

from argparse import ArgumentParser

def decode_voc(root_xml):
    size_xml = root_xml.find('size')
    objects = []

    for object_xml in root_xml.findall('object'):
        label = object_xml.find('name').text
        box = object_xml.find('bndbox')

        objects.append((label, (
            float(box.find('xmin').text),
            float(box.find('ymin').text),
            float(box.find('xmax').text),
            float(box.find('ymax').text) 
        )))

    return {
        'image_path': root_xml.find('path').text,
        'image_width': float(size_xml.find('width').text),
        'image_height': float(size_xml.find('height').text),
        'objects': objects
    }

def voc_to_darknet(cls, box, image_width, image_height):
    # convert to darknet's format:
    # <class> <x_center> <y_center> <width> <height>
    # where all points/sizes are relative to the image width
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + (w / 2)
    cy = box[1] + (h / 2)
    
    return '%d %.6f %.6f %.6f %.6f' % (cls, cx / image_width, cy / image_height, w / image_width, h / image_height)

def convert_annotations(base_dir, species_list):
    image_dir = os.path.abspath(os.path.join(base_dir, 'images'))

    with open(os.path.join(base_dir, 'darknet.data'), 'w') as out_file:
        with os.scandir(os.path.join(base_dir, 'annotations')) as scanner:
            for entry in scanner:
                filename, ext = os.path.splitext(entry.name)
                if not entry.is_file() or ext != '.xml':
                    continue

                # read the existing VOC file and convert it to a simple txt equivalent
                voc = decode_voc(ET.parse(entry.path))
                im_width = voc['image_width']
                im_height = voc['image_height']
                objects = [ voc_to_darknet(species_list.index(label), box, im_width, im_height) for label, box in voc['objects'] ]
                darknet_path = os.path.join(image_dir, filename + '.txt')

                with open(darknet_path, 'w') as txt_file:
                    txt_file.write('\n'.join(objects))

                out_file.write(os.path.join(image_dir, os.path.basename(voc['image_path'])))
                out_file.write('\n')

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
