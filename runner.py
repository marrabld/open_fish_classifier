import json
import os
import re

from cv2 import cv2
from track import track
from argparse import ArgumentParser

VIDEO_DIRECTORY = r"E:\Work\data\fish\videos"
BOUNDING_BOXES = r".\bounding-boxes.json"

def shapeToBox(shape):
    return (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))

if __name__ == '__main__':
    parser = ArgumentParser(description='Attempt to track bounding boxes through frames')
    parser.add_argument('attempt', help='Name of the attempt')

    # argparse terminates the process if parse_args() encounters an error
    args = parser.parse_args()
    metadata = None

    with open(BOUNDING_BOXES, 'r') as fd:
        metadata = json.load(fd)

    for meta in metadata:
        match = re.search(r'^(?P<video>.*?)(?P<frame>\d+)\.\w+$', meta['filename'])

        if match is None:
            print('unable to parse filename "%s"'%meta['filename'])
            continue

        infile = os.path.join(VIDEO_DIRECTORY, match['video'])
        startFrame = int(match['frame'])

        if not os.path.exists(infile):
            print('unable to find "%s" on disk'%match['video'])
            continue

        # convert shapes to box tuples
        boxes = [ shapeToBox(r['shape_attributes']) for r in meta['regions'] ]

        # initialise the reader
        video = cv2.VideoCapture(infile)
        video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        # initialise the writer to match the input video dimensions / FPS
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        outfile = './mov/out/%s_%s.%d.avi'%(os.path.splitext(match['video'])[0], args.attempt, startFrame)
        writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), int(fps / 4), (width, height))

        track(video, writer, boxes, 200)
