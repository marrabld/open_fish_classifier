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
            float(box.find('xmin').text),
            float(box.find('ymin').text),
            float(box.find('xmax').text),
            float(box.find('ymax').text) 
        )))

    return image_path, objects

def select_label(prediction, labels):
    ranked = np.argsort(prediction.classes)
    top_label = labels[ranked[-1]]

    if top_label == 'fish' and len(ranked) > 1 and prediction.classes[ranked[-2]] > 0.8:
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

def compute_iou_matrix(boxes_a, boxes_b):
    iou_matrix = np.empty((len(boxes_a), len(boxes_b)), dtype='float')

    bx1 = boxes_b[:,0]
    by1 = boxes_b[:,1]
    bx2 = boxes_b[:,2]
    by2 = boxes_b[:,3]

    areas_a = (boxes_a[:,2] - boxes_a[:,0] + 1) * (boxes_a[:,3] - boxes_a[:,1] + 1)
    areas_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)

    for i, tb in enumerate(boxes_a):
        x1 = np.maximum(tb[0], bx1)
        y1 = np.maximum(tb[1], by1)
        x2 = np.minimum(tb[2], bx2)
        y2 = np.minimum(tb[3], by2)

        intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        union = areas_b - intersection + areas_a[i]
        iou_matrix[i:] = intersection / union

    return iou_matrix

def select_label(prediction, labels):
    ranked = np.argsort(prediction.classes)
    top_label = labels[ranked[-1]]

    if top_label == 'fish' and prediction.classes[ranked[-2]] > 0.8:
        top_label = labels[ranked[-2]]
        
    return top_label

def is_correct_label(correct_label, prediction, all_labels):
    ranked = np.argsort(prediction.classes)
    top_label = all_labels[ranked[-1]]

    return top_label == correct_label or (top_label == 'fish' and prediction.classes[ranked[-2]] > 0.75 and all_labels[ranked[-2]] == correct_label)

def main(args):
    root_dir = os.path.join('training', args.training)

    if not os.path.isdir(root_dir):
        print('error: unable to find training run named "%s"' % args.training)
        return 1

    output_dir = os.path.join(root_dir, 'f1')
    config_path = os.path.join(root_dir, 'json', 'detection_config.json')
    weights_path = find_weights_path(root_dir, args.epoch)

    if weights_path is None:
        print('error: unknown epoch requested "%d"' % args.epoch)
        return 1

    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # initialise the detector
    from object_detector import ObjectDetector
    detector = ObjectDetector.from_config(weights_path, config_path)

    labelled_true_positives = 0
    unlabelled_true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_objects = 0

    with os.scandir(os.path.join(root_dir, 'test', 'train', 'annotations')) as scanner:
        for entry in scanner:
            if not entry.is_file() or not entry.name.endswith('.xml'):
                continue

            tree = ET.parse(entry.path)
            image_path, objects = decode_voc_objects(tree.getroot())
            image = cv2.imread(image_path)

            # use non-maximum supression to de-duplicate similar bounding boxes
            predictions = detector.detect(image, args.object_threshold, args.nms_threshold)
            total_objects += len(objects)

            if len(objects) == 0:
                # everything is a false positive
                false_positives += len(predictions)
                continue

            # compute iou matrix between predictions and objects
            truth_boxes = np.array([ points for _, points in objects ], dtype='float')
            pred_boxes = np.array([ [p.xmin, p.ymin, p.xmax, p.ymax ] for p in predictions ], dtype='float')
            iou_matrix = compute_iou_matrix(truth_boxes, pred_boxes)

            # categorise all truth objects as either true positives or false negatives
            for i, tb in enumerate(truth_boxes):
                max_index = np.argmax(iou_matrix[i,])
                prediction = predictions[max_index]
                label, _ = objects[i]

                # draw the ground truth first
                top_left, bottom_right = ((int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])))

                if iou_matrix[i, max_index] < args.iou_threshold:
                    false_negatives += 1
                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 2)
                else:
                    # clear out the IoU column for this prediction so it is not re-used
                    iou_matrix[:, max_index] = 0

                    if is_correct_label(label, prediction, detector.labels):
                        labelled_true_positives +=1
                        color = (0, 255, 0) # green
                    else:
                        unlabelled_true_positives += 1
                        color = (180, 44, 255) # pink

                    pb = pred_boxes[max_index]
                    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)
                    cv2.rectangle(image, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), color, 2)
                    #cv2.putText(image, str(i), (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
                    cv2.putText(image, str(i), (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, color)
                    #cv2.putText(image, str(i), (int(pb[0]), int(pb[1] - 5)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
                    cv2.putText(image, str(i), (int(pb[0]), int(pb[1] - 5)), cv2.FONT_HERSHEY_PLAIN, 1, color)

            # any columns that still have an IoU value were not used for any object detection
            # so just sum the number of columns with any value > 0 to compute false positives
            #false_positives = len(predictions) - np.count_nonzero(iou_matrix, axis=0)
            fp_indices = np.where(np.amax(iou_matrix, axis=0) > 0)[0]
            false_positives += len(fp_indices)

            for index in fp_indices:
                pb = pred_boxes[index]
                cv2.rectangle(image, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (0, 0, 255), 2)

            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image)

    true_positives = unlabelled_true_positives + labelled_true_positives
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accuracy = labelled_true_positives / true_positives

    print('recall:          %.4f' % recall)
    print('precision:       %.4f' % precision)
    print('accuracy:        %.4f' % accuracy)
    print('f1:              %.4f' % (2 * ((precision * recall) / (precision + recall))))

    print('true positives:  %d (%.2f%%)' % (true_positives, 100 * (true_positives / float(total_objects))))
    print('false positives: %d (%.2f%%)' % (false_positives, 100 * (false_positives / float(true_positives + false_positives))))
    print('false negatives: %d (%.2f%%)' % (false_negatives, 100 * (false_negatives / float(total_objects))))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('f1', description='Compute a F1 score for a YOLO training run')
    parser.add_argument('-t', '--training', required=True, help='Name of the training run to compute the F1 score for')
    parser.add_argument('-e', '--epoch', required=False, help='The epoch of a specific model to use to compute the F1 score. Defaults to the latest epoch', default=None)
    parser.add_argument('-i', '--iou-threshold', required=False, type=float, help='Minimum Intersection of Union required to consider a prediction a match', default=0.5)
    parser.add_argument('-o', '--object-threshold', required=False, type=float, help='Minimum "object-ness" required to consider a YOLO detection valid', default=0.5)
    parser.add_argument('-n', '--nms-threshold', required=False, type=float, help='Non-Maximum Supression threshold used to de-duplicate similar detections', default=0.4)
    parser.add_argument('-d', '--output-directory', required=False, help='Output directory to save snapshot images to', default=None)

    args = parser.parse_args()
    exit(main(args) or 0)
