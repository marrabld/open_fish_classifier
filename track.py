import numpy as np

from random import randint
from argparse import ArgumentParser
from cv2 import cv2

TEMPLATE_THRESHOLD = 0.6
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

def track(reader, writer, boxes, totalFrames):
    bgsub = cv2.createBackgroundSubtractorKNN()

    for _ in range(totalFrames):
        res, colorFrame = reader.read()

        if not res:
            print('ran out of frames')
            break

        fgmask = createMask(colorFrame, bgsub, -1)
        boxes = computeBoundingBoxes(fgmask)

        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        output = cv2.addWeighted(fgmask, 0.5, colorFrame, 0.5, 0)
        drawBoundingBoxes(output, boxes)

        writer.write(output)

def computeBoundingBoxes(mask, minArea=64.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [ cv2.boundingRect(c) for c in contours ]

    # get around a 'bug' or 'feature' of groupRectangles where it removes standalone rectangles
    # (threshold = 0 is a no-op, but threshold = 1 discards rectangles that don't have any sub-rectangles)
    # https://stackoverflow.com/questions/21421070/opencv-grouprectangles-getting-grouped-and-ungrouped-rectangles
    #partitioned, _ = cv2.groupRectangles(rectangles + rectangles, 1, 0.5)
    return [ r for r in rectangles if (r[2] * r[3]) >= minArea ]

def createMask(image, bgsub, learn):
    mask = bgsub.apply(image, learn)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    return mask

def drawBoundingBoxes(frame, boxes):
    for rect in boxes:
        cv2.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), rrgb(), 2)

def rrgb():
    return (randint(0, 255), randint(0, 255), randint(0, 255))