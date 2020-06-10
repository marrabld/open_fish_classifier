import cv2
import pickle
import os
import numpy as np
import time

from argparse import ArgumentParser
from object_detector import ObjectDetector

def get_top_label(detection, labels):
    ranked = np.argsort(detection.classes)
    top_index = ranked[-1]

    if labels[top_index] == 'fish' and detection.classes[ranked[-2]] > 0.6:
        top_index = ranked[-2]

    return labels[top_index], detection.classes[top_index]

def detect(video_path, detector, stride):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // stride
    frame_numbers = [ None ] * total_frames
    frame_detections = [ None ] * total_frames

    start_time = time.time()

    for frame_index in range(total_frames):
        frame_number = frame_index * stride

        if stride > 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        res, frame = video.read()

        if not res:
            print('warning: ran out of frames prematurely')
            break

        if (frame_index % 100) == 0:
            print('info: processed up to frame %d' % frame_number)

        detections = detector.detect(frame, 0.3, 0.5)
        frame_numbers[frame_index] = frame_number
        frame_detections[frame_index] = [ ( *get_top_label(d, detector.labels), d.xmin, d.ymin, d.xmax, d.ymax ) for d in detections ]

    elapsed = time.time() - start_time # seconds
    print('info: completed in %.2fs, average fps: %.2f' % (elapsed, total_frames / elapsed))

    return frame_numbers, frame_detections

def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    detector = ObjectDetector.from_config(args.model_path, args.config_path)
    frames = detect(args.video_path, detector, args.stride)

    with open(args.output_path, 'wb') as output_file:
        pickle.dump(frames, output_file)

if __name__ == '__main__':
    parser = ArgumentParser('gen_detections', description='Using ImageAI and YOLOv3, generate frame-level detections for a video, saving them in Python\'s "pickle" format')
    parser.add_argument('-m', '--model-path', required=True, help='Path to YOLOv3 model to use for detection')
    parser.add_argument('-c', '--config-path', required=True, help='Path to ImageAI JSON configuration file')
    parser.add_argument('-s', '--stride', type=int, required=False, default=1, help='Number of frames to progress between each detection')
    parser.add_argument('-o', '--output-path', required=True, help='Path to write the detections to')
    parser.add_argument('video_path', help='Path to input video file')

    exit(main(parser.parse_args()) or 0)
