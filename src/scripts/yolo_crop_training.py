import os
import shutil
import cv2

from argparse import ArgumentParser

DATA_DIRECTORIES = [ 'train', 'validation', 'test' ]

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
</annotation>
"""

def gen_annotation(label, path, width, height):
    return ANNOTATION_FORMAT.format(
        folder = os.path.dirname(path),
        filename = os.path.basename(path),
        path = path,
        width = width,
        height = height,
        label = label,
        xmin = 0.0,
        ymin = 0.0,
        xmax = width,
        ymax = height
    )

def process_images(species, image_dir, output_dir):
    # walk the image directory
    with os.scandir(image_dir) as scanner:
        for entry in scanner:
            if entry.is_file():
                img = cv2.imread(entry.path)

                # ignore any non-image files
                if img is not None:
                    annotation_path = os.path.join(output_dir, 'annotations', os.path.splitext(entry.name)[0] + '.xml')
                    image_path = os.path.join(output_dir, 'images', entry.name)

                    with open(annotation_path, 'w') as af:
                        af.write(gen_annotation(species, image_path, img.shape[1], img.shape[0]))

                    os.link(entry.path, image_path)
                    
def make_output_dirs(root):
    for dirname in DATA_DIRECTORIES:
        os.makedirs(os.path.join(root, dirname, 'images'), exist_ok=False)
        os.makedirs(os.path.join(root, dirname, 'annotations'), exist_ok=False)

def train_model(root_dir, species, batch_size, pretrained_path):
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=root_dir)

    if pretrained_path is not None:
        trainer.setTrainConfig(object_names_array=species, batch_size=batch_size, train_from_pretrained_model=pretrained_path)
    else:
        trainer.setTrainConfig(object_names_array=species, batch_size=batch_size)

    trainer.trainModel()

def main(args):
    # Sanity check the input directory
    if not os.path.exists(args.crop_directory):
        print('error: unable to find directory "%s"' % args.crop_directory)
        return 1

    root_dir = os.path.abspath(os.path.join('output', args.name))

    if os.path.exists(root_dir):
        if not args.force_overwrite:
            print('error: a "%s" training run already exists, specify "--force-overwrite" or choose a unique name' % args.name)
            return 1

        # remove the old training data
        print('warning: overwriting previous "%s" training run' % args.name)
        shutil.rmtree(root_dir)

    # create the output directory structure, it should not exist at this point
    make_output_dirs(root_dir)
    print('info: generating training dataset')

    # process each species, using the directory == species naming convention
    for species in args.species:
        for subdir in DATA_DIRECTORIES:
            image_dir = os.path.join(args.crop_directory, species, subdir)
            output_dir = os.path.join(root_dir, subdir)

            if not os.path.exists(image_dir):
                print('error: unable to find directory "%s"' % image_dir)
                return 1

            process_images(species, image_dir, output_dir)

    # train the model
    print('info: training model')
    train_model(root_dir, args.species, args.batch_size, args.pretrained)

    return 0

if __name__ == '__main__':
    parser = ArgumentParser('train', description='Train a YOLOv3 model from crop data')
    parser.add_argument('--crop-directory', required=True, help='Root directory containing all of the crop data partitioned into train/validate/test folders')
    parser.add_argument('-s', '--species', required=True, action='append', help='Species to train the model on (subset of crop data, can be specified multiple times)')
    parser.add_argument('-p', '--pretrained', required=False, help='Path to pre-trained YOLO model to apply transfer learning from', default=None)
    parser.add_argument('-f', '--force-overwrite', required=False, action='store_true', help='Force overwrite a previous training run with the same name', default=False)
    parser.add_argument('-b', '--batch-size', required=False, type=int, help='Batch size for model training (default: 4)', default=4)
    parser.add_argument('name', help='Name of the training run, must be unique unless "--force-overwrite" was specified')

    args = parser.parse_args()
    exit(main(args) or 0)