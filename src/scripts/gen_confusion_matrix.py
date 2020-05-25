import cv2
import numpy as np
import json
import re
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# hardcoded input image size to use with YOLO (hardcoded in ImageAI)
INPUT_SIZE = 416

def decode_voc_objects(root_xml):
    image_path = root_xml.find('path').text
    objects = []

    for object_xml in root_xml.findall('object'):
        label = object_xml.find('name').text
        box = object_xml.find('bndbox')

        objects.append((label, (
            int(float(box.find('xmin').text)),
            int(float(box.find('ymin').text)),
            int(float(box.find('xmax').text)),
            int(float(box.find('ymax').text)) 
        )))

    return image_path, objects

def compute_predictions(raw_image, model, anchors, utils, threshold):
    height, width, _ = raw_image.shape

    image = cv2.resize(raw_image, (INPUT_SIZE, INPUT_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.
    # expand the image to batch
    image = np.expand_dims(image, 0)

    predictions = model.predict(image)
    boxes = []

    for idx, [ prediction ] in enumerate(predictions):
        boxes += utils.decode_netout(prediction, anchors[idx], threshold, INPUT_SIZE, INPUT_SIZE)

    utils.correct_yolo_boxes(boxes, height, width, INPUT_SIZE, INPUT_SIZE)
    return boxes
    #return [ (b.classes, (b.xmin, b.ymin, b.xmax, b.ymax)) for b in boxes ]

def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    i_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    u_area = a_area + b_area - i_area

    return i_area / float(u_area)

def select_prediction(truth_box, predictions, threshold):
    ious = np.array([ iou(truth_box, (p.xmin, p.ymin, p.xmax, p.ymax)) for p in predictions ])
    top = np.argmax(ious)
    return predictions[top] if ious[top] >= 0.6 else None

def select_label(prediction, labels):
    ranked = np.argsort(prediction.classes)
    top_label = labels[ranked[-1]]

    if top_label == 'fish' and prediction.classes[ranked[-2]] > 0.8:
        top_label = labels[ranked[-2]]
        
    return top_label

def find_weights_path(root_dir, epoch):
    model_dir = os.path.join(root_dir, 'models')
    all_paths = os.listdir(model_dir)
    path = os.path.join(model_dir, all_paths[-1]) # default to last in list, assumed to be final epoch snapshot

    if epoch is not None:
        pattern = r'^detection_model-ex-%03d--loss-\d+\.\d+.h5$' % epoch
        path = next((os.path.join(model_dir, n) for n in all_paths if re.match(pattern, n)), None)
    
    return path

def main(args):
    root_dir = os.path.join('training', args.training)

    if not os.path.isdir(root_dir):
        print('error: unable to find training run named "%s"' % args.training)
        return 1

    with open(os.path.join(root_dir, 'json', 'detection_config.json')) as config_file:
        config = json.load(config_file)
        labels = config['labels']
        anchors = config['anchors']

    weights_path = find_weights_path(root_dir, args.epoch)

    if weights_path is None:
        print('error: unknown epoch requested "%d"' % args.epoch)
        return 1

    from imageai.Detection.YOLOv3.models import yolo_main
    from imageai.Detection.Custom import CustomDetectionUtils, CustomObjectDetection
    from keras import Input

    utils = CustomDetectionUtils(labels)
    model = yolo_main(Input(shape=(None, None, 3)), 3, len(labels))
    model.load_weights(weights_path)

    with os.scandir(os.path.join(root_dir, 'test', 'train', 'annotations')) as scanner:
        truth_labels = []
        pred_labels = []  

        for entry in scanner:
            if not entry.is_file() or not entry.name.endswith('.xml'):
                continue

            # decode the annotation file
            xml = ET.parse(entry.path)
            image_path, objects = decode_voc_objects(xml.getroot())

            # read the image and compute the predictions
            image = cv2.imread(image_path)
            predictions = compute_predictions(image, model, anchors, utils, args.object_threshold)

            # for each object in the file, select the best* prediction
            for label, box in objects:
                best = select_prediction(box, predictions, args.iou_threshold)
                
                if best is not None:
                    truth_labels.append(label)
                    pred_labels.append(select_label(best, labels))
                elif not args.exclude_undetected:
                    truth_labels.append(label)
                    pred_labels.append('-')

        matrix_labels = labels if args.exclude_undetected else [ *labels, '-' ]
        matrix = confusion_matrix(truth_labels, pred_labels, normalize='true', labels=matrix_labels)
        display = ConfusionMatrixDisplay(matrix, display_labels=matrix_labels)

        plot = display.plot(cmap='Blues', xticks_rotation='45')
        figure = plt.gcf()
        figure.set_size_inches(18, 18)
        plt.savefig(args.output_image_path, bbox_inches='tight')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('gen_confusion_matrix', description='Generate a confusion matrix for a training run')
    parser.add_argument('-t', '--training', required=True, help='Name of the training run to generate the matrix from')
    parser.add_argument('-e', '--epoch', required=False, help='The epoch of a specific model to use to generate the matrix, defaults to the latest epoch', default=None)
    parser.add_argument('-i', '--iou-threshold', required=False, type=float, help='Minimum Intersection of Union required to consider a prediction a match', default=0.6)
    parser.add_argument('-o', '--object-threshold', required=False, type=float, help='Minimum "object-ness" required to consider a YOLO detection valid', default=0.5)
    parser.add_argument('-x', '--exclude-undetected', required=False, action='store_true', help='Flag to exclude objects that had no predictions from the matrix', default=False)
    parser.add_argument('output_image_path', help='File to output the confusion matrix plot to')

    args = parser.parse_args()
    exit(main(args) or 0)
