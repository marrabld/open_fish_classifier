import json
import re
import os
import numpy

from random import randint
from argparse import ArgumentParser
from cv2 import cv2

TEMPLATE_THRESHOLD = 0.6
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

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

def drawPoints(frame, points):
    print(len(points))
    for point in points:
        cv2.circle(frame, (point[0][0], point[0][1]), 4, (255, 0, 0), 2)
    return frame

def computeBoundingBoxes(mask, minArea=64.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [ cv2.boundingRect(c) for c in contours ]

    # get around a 'bug' or 'feature' of groupRectangles where it removes standalone rectangles
    # (threshold = 0 is a no-op, but threshold = 1 discards rectangles that don't have any sub-rectangles)
    # https://stackoverflow.com/questions/21421070/opencv-grouprectangles-getting-grouped-and-ungrouped-rectangles
    #partitioned, _ = cv2.groupRectangles(rectangles + rectangles, 1, 0.5)
    return [ r for r in rectangles if (r[2] * r[3]) >= minArea ]

def logBoundingBoxes(writer, grayscale, boxes):
    color = grayscale # cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    
    for rect in boxes:
        cv2.rectangle(color, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2)

    writer.write(color)

def createMask(image, bgsub, learn):
    mask = bgsub.apply(image, learn)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    return mask

def nextStateMT(frame, current):
    nextState = []

    for state in current:
        id, template, lastBox, color = state
        _, maxVal, _, maxLoc = cv2.minMaxLoc(cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED))

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
    parser.add_argument('--seed-frame', type=int, help='Video frame that contains mostly background', required=True)
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

    # load up the video and seed the background subtraction
    video = cv2.VideoCapture(args.video)
    res, seedFrame = readFrame(video, args.seed_frame)
    bgsub = cv2.createBackgroundSubtractorKNN()
    bgsub.apply(seedFrame, 1)

    # initialise the output writer with the same dimensions as the input
    res, frame = readFrame(video, args.start_frame)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), video.get(cv2.CAP_PROP_FPS), frame.shape[::-1][1:])
    
    # draw first frame
    fgmask = createMask(frame, bgsub, -1)
    boxes = computeBoundingBoxes(fgmask)
    logBoundingBoxes(writer, frame, boxes)
    #fgmask = bgsub.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, MORPH_KERNEL)
    #fgmask = cv2.threshold(fgmask, 1, 255, cv2.THRESH_BINARY)[1]
    #colorCopy = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    #overlayBoundingBoxes(colorCopy, createMask(frame, bgsub, 0))

    #cv2.imwrite('./mov/sub/%s_%d.png'%(os.path.basename(args.output), args.start_frame), colorCopy)


    # initialise the tracker state with the initial box locations and templates
    state = [ (i, frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], b, rrgb()) for i, b in enumerate(boxes) ]
    # writeTrackerState(writer, frame, state)

    # start stepping through the frames
    for i in (range(args.total_frames)):
        res, frame = video.read()

        # bail early if unable to read the next frame
        # TODO: Actual error handling instead of assuming it's because of EOF
        if not res:
            break

        fgmask = createMask(frame, bgsub, -1)
        boxes = computeBoundingBoxes(fgmask)
        logBoundingBoxes(writer, frame, boxes)

        #fgmask = bgsub.apply(frame, 0)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, MORPH_KERNEL)
        #fgmask = cv2.threshold(fgmask, 1, 255, cv2.THRESH_BINARY)[1]
        #colorCopy = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        #cv2.imwrite('./mov/sub/%s_%d.png'%(os.path.basename(args.output), args.start_frame + i), colorCopy)

        #overlayBoundingBoxes(colorCopy, fgmask)
        #writer.write(colorCopy)

        #writer.write(drawPoints(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR), cv2.goodFeaturesToTrack(grayFrame, 100, 0.3, 7, mask=fgmask)))

        # compute the next state from the frame and the current state
        #state = nextStateMT(frame, state)
        #writeTrackerState(writer, frame, state)

    writer.release()
    video.release()
    cv2.destroyAllWindows()
    
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