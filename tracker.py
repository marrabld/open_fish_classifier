import json
import re
import os
import numpy

from random import randint
from argparse import ArgumentParser
from cv2 import cv2

TEMPLATE_THRESHOLD = 0.6

def rrgb():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def readFrame(video, frame):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    return video.read()

def shapeToBox(shape):
    x = int(shape['x'])
    y = int(shape['y'])
    return (x, y, x + int(shape['width']), y + int(shape['height']))

def pushTrackerState(output, frame, tracker):
    for tracked in tracker:
        _, _, box, color = tracked
        cv2.rectangle(frame, box[0:2], box[2:4], color, 3)

    output.write(frame)

if __name__ == '__main__':
    parser = ArgumentParser(description='Attempt to track bounding boxes through frames')
    parser.add_argument('input', help='Input JSON file describing bounding boxes and labels for fish')
    parser.add_argument('--video-directory', help='Directory containing video files referenced in input', required=True)

    # argparse terminates the process if parse_args() encounters an error
    args = parser.parse_args()
    files = None
    videoDirectory = args.video_directory

    with open(args.input) as f:
        files = json.load(f)

    for meta in files:
        # Split apart filename into video/frame
        match = re.search(r'^(?P<video>.*?)(?P<frame>\d+)\.\w+$', meta['filename'])
        
        if match is None:
            print('unable to parse "%s" as video/frame combination'%(meta['filename']))
            continue

        # Try and locate the video in the video directory
        path = os.path.join(videoDirectory, match['video'])

        if not os.path.exists(path):
            print('unable to find "%s" in video directory, skipping'%(match['video']))
            continue

        # Load the video and initialise the tracker
        tracker = []
        fnum = int(match['frame'])
        video = cv2.VideoCapture(path)
        res, frame = readFrame(video, fnum)
        writer = cv2.VideoWriter('./mov/templates/%s.%d.template.avi'%(os.path.splitext(match['video'])[0], fnum), cv2.VideoWriter_fourcc(*'DIVX'), 15, frame.shape[::-1][1:])

        # TODO: test if it's worth converting the templates to grayscale
        for region in meta['regions']:
            box = shapeToBox(region['shape_attributes'])
            template = frame[box[1]:box[3], box[0]:box[2]]
            tracker.append((region['region_attributes']['label'], template, box, rrgb()))

        pushTrackerState(writer, frame, tracker)

        # Capture another 100 frames (or exit when no tracking objects are left)
        for i in range(100):
            fnum += 1
            res, frame = video.read()
            updated = []

            if not res:
                break

            for tracked in tracker:
                label, template, lastBox, color = tracked

                _, maxVal, _, maxLoc = cv2.minMaxLoc(cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED))

                if maxVal > TEMPLATE_THRESHOLD:
                    nextBox = (maxLoc[0], maxLoc[1], maxLoc[0] + (lastBox[2] - lastBox[0]), maxLoc[1] + (lastBox[3] - lastBox[1]))
                    updated.append((label, frame[nextBox[1]:nextBox[3], nextBox[0]:nextBox[2]].copy(), nextBox, color))

            if len(updated) == 0:
                break

            tracker = updated    
            pushTrackerState(writer, frame, tracker)

        writer.release()
        video.release()
        
