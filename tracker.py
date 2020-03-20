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
    return (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))

def writeTrackerState(writer, frame, current):
    for state in current:
        id, _, box, color = state
        cv2.rectangle(frame, box[0:2], (box[0] + box[2], box[1] + box[3]), color, 3)
        cv2.putText(frame, str(id), box[0:2], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    writer.write(frame)

def nextStateMT(frame, current):
    nextState = []

    for state in current:
        id, template, lastBox, color = state
        _, maxVal, _, maxLoc = cv2.minMaxLoc(cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED))

        if id == 4:
            print(maxVal)

        if maxVal >= TEMPLATE_THRESHOLD:
            nextBox = (maxLoc[0], maxLoc[1], lastBox[2], lastBox[3])
            nextTemplate = frame[nextBox[1]:nextBox[1]+nextBox[3], nextBox[0]:nextBox[0]+nextBox[2]].copy()
            nextState.append((id, nextTemplate, nextBox, color))

    return nextState

if __name__ == '__main__':
    parser = ArgumentParser(description='Attempt to track bounding boxes through frames')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('--regions', help='JSON file containing region information', required=True)
    parser.add_argument('--start-frame', type=int, help='Video frame that the bounding boxes apply to', required=True)
    parser.add_argument('--total-frames', type=int, help='Total number of frames to track through', required=False, default=100)
    parser.add_argument('--output', help='Path to save the resulting clip to', required=True)

    # argparse terminates the process if parse_args() encounters an error
    args = parser.parse_args()

    # ensure that the output directory exists
    outputDir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # load the boxes into memory
    boxes = None

    with open(args.regions, 'r') as f:
        regions = json.load(f)
        boxes = [ shapeToBox(r['shape_attributes']) for r in regions ]

    # load up the video and capture the first frame
    fnum = args.start_frame
    video = cv2.VideoCapture(args.video)
    res, frame = readFrame(video, fnum)

    # initialise the output writer with the same dimensions as the input
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'DIVX'), 15, frame.shape[::-1][1:])

    # initialise the tracker state with the initial box locations and templates
    state = [ (i, frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], b, rrgb()) for i, b in enumerate(boxes) ]
    writeTrackerState(writer, frame, state)

    # start stepping through the frames
    for i in (range(args.total_frames)):
        res, frame = video.read()

        # bail early if unable to read the next frame
        # TODO: Actual error handling instead of assuming it's because of EOF
        if not res:
            break

        # compute the next state from the frame and the current state
        state = nextStateMT(frame, state)
        writeTrackerState(writer, frame, state)

    writer.release()
    video.release()
    
"""
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
        
"""