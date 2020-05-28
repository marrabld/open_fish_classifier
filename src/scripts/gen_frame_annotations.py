import os
import cv2
import csv
import re
import numpy as np

from object_detector import ObjectDetector
from argparse import ArgumentParser

def normalize_header(header):
    return re.sub('[^\w]+', '_', header.lower())

def get_or_set(dictionary, key, generator):
    value = dictionary.get(key, None)

    if value is None:
        value = generator()
        dictionary[key] = value

    return value

def get_top_label(detection, labels):
    ranked = np.argsort(detection.classes)
    top_index = ranked[-1]

    if labels[top_index] == 'fish' and detection.classes[ranked[-2]] > 0.6:
        top_index = ranked[-2]

    return labels[top_index], detection.classes[top_index]

def get_label_color(label_index):
    color_wheel = [ 
        (0xFF, 0xFF, 0xFF), 
        (0x77, 0x66, 0xCC),
        (0x77, 0xCC, 0xDD),
        (0x33, 0x77, 0x11),
        (0x00, 0x00, 0xFF),
        (0x99, 0x44, 0xAA),
        (0x00, 0xFF, 0x00),
        (0x33, 0x99, 0x99),
        (0x55, 0x22, 0x88),
        (0x00, 0x11, 0x66),
        (0x00, 0xFF, 0xFF),
        (0x88, 0x88, 0x88)
    ]

    return color_wheel[label_index % len(color_wheel)]

def parse_points_csv(csv_path):
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        columns = { normalize_header(headers[i]): i for i in range(len(headers)) }
        results = dict()

        col_file_name = columns['filename']
        col_frame = columns['frame']
        col_x = columns['image_col']
        col_y = columns['image_row']
        col_family = columns['family']
        col_genus = columns['genus']
        col_species = columns['species']

        for row in reader:
            file_name = row[col_file_name]
            frame = int(row[col_frame])
            x = int(float(row[col_x]))
            y = int(float(row[col_y]))
            label = '%s_%s' % (row[col_genus], row[col_species])

            if label == '_':
                continue
            
            file = get_or_set(results, file_name, lambda: dict())
            annotations = get_or_set(file, frame, lambda: [])
            
            annotations.append((label.lower(), x, y))

    return results


def main(args):
    detector = ObjectDetector.from_config(args.model_path, args.config_path)
    annotations = parse_points_csv(args.csv_path)

    # reusable colors
    white = (255, 255, 255)

    os.makedirs(args.output_directory, exist_ok=True)

    for file_name in annotations:
        name, _ = os.path.splitext(file_name)
        reader = cv2.VideoCapture(os.path.join(args.video_directory, file_name))

        if not reader.isOpened():
            print('warning: unable to find "%s" in video directory, skipping' % file_name)
            continue

        file_annotations = annotations[file_name]

        for frame_number in file_annotations:
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            res, frame = reader.read()

            if not res:
                print('warning: unable to read frame %d from "%s"' (frame_number, file_name))
                continue

            # do our own detections
            detections = detector.detect(frame, object_threshold=0.5, nms_threshold=0.5)

            for detection in detections:
                label, confidence = get_top_label(detection, detector.labels)
                label_color = get_label_color(detector.labels.index(label))

                cv2.rectangle(frame, (detection.xmin, detection.ymin), (detection.xmax, detection.ymax), label_color, 1)
                cv2.putText(frame, '%s: %.3f' % (label, confidence), (detection.xmin, detection.ymin - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, white, 1)

            # write the image with just our detections
            cv2.imwrite(os.path.join(args.output_directory, '%s.%d.detected.png' % (name, frame_number)), frame)

            # add the annotations
            frame_annotations = file_annotations[frame_number]

            for label, x, y in frame_annotations:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), 4)
                cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_PLAIN, 0.75, white, 1)

            cv2.imwrite(os.path.join(args.output_directory, '%s.%d.detected.annotated.png' % (name, frame_number)), frame)

        reader.release()
    
if __name__ == '__main__':
    parser = ArgumentParser('gen_frame_annotations', description='Generates a dataset of comparison images between YOLO detections and hand annotations')
    parser.add_argument('-o', '--output-directory', required=True)
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-c', '--config-path', required=True)
    parser.add_argument('-f', '--csv-path', required=True)
    parser.add_argument('-v', '--video-directory', required=True)

    exit(main(parser.parse_args()) or 0)