import numpy as np

from random import randint
from argparse import ArgumentParser
from cv2 import cv2

TEMPLATE_THRESHOLD = 0.6
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

def track(reader, writer, boxes, totalFrames):
    _, colorFrame = reader.read()
    frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

    trackers = [ Tracker(b, sampleFeatures(frame, b)) for b in boxes ]
    lastPoints = np.concatenate(tuple([ t.samples for t in trackers if t.samples is not None ]))
    lastFrame = frame

    # Tweak-able LK parameters
    lkParams = dict(
        winSize = (15, 15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    for _ in range(totalFrames):
        res, colorFrame = reader.read()

        if not res:
            print('ran out of frames')
            break

        frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

        points, status, _ = cv2.calcOpticalFlowPyrLK(lastFrame, frame, lastPoints, None, **lkParams)
        drawPoints(colorFrame, points)

        pos = 0
        for tracker in trackers:
            if tracker.samples is None:
                continue

            ns = len(tracker.samples)
            tracker.update(frame, points[pos:pos+ns], status[pos:pos+ns])
            pos += ns

            x, y, w, h = tracker.bounds
            cv2.rectangle(colorFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        lastPoints = np.concatenate(tuple([ t.samples for t in trackers if t.samples is not None ]))
        writer.write(colorFrame)
        
        lastFrame = frame

class Tracker:
    def __init__(self, bounds, samples):
        self.samples = samples
        self.bounds = bounds

    def update(self, grayscale, samples, statuses):
        # compute vectors
        assert(len(samples) == len(self.samples))
        vectors = np.subtract(samples, self.samples)
        average = np.mean(vectors, axis=0)

        x, y, w, h = self.bounds
        x = int(x + average[0])
        y = int(y + average[1])

        # clamp values
        if x < 0:
            w = w + x
            x = 0
        
        if y < 0:
            h = h + y
            y = 0

        self.bounds = (x, y, w, h)

        # If any points are now outside the new bounding box, resample the features
        # TODO: Add some sort of threshold as the bounding box size never changes
        # TODO: Worth preserving the in-bounds samples and just resampling the ones that have been lost?
        if not all(isInRectangle(pt, self.bounds) for pt in samples):
            self.samples = sampleFeatures(grayscale, self.bounds)
        else:
            self.samples = samples

        return self.samples

def drawPoints(frame, points):
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

def sampleFeatures(grayscale, bounds):
    x, y, w, h = bounds
    
    # Avoid trying to sample tiny boxes at all
    if w <= 5 or h <= 5:
        return None 

    # TODO: Make these configurable 
    featureParams = dict(
        maxCorners = 10,
        qualityLevel = 0.02,
        minDistance = 0.02 * ((w + h) / 2)
    )

    crop = grayscale[y:y+h, x:x+w]
    corners = cv2.goodFeaturesToTrack(crop, **featureParams)

    if corners is None:
        return None

    # Normalise the points back to the parent image
    return np.array([ [ x + c[0][0], y + c[0][1] ] for c in corners ], dtype=np.float32)

def isInRectangle(point, rectangle):
    return point[0] >= rectangle[0] \
        and point[0] < (rectangle[0] + rectangle[2]) \
        and point[1] >= rectangle[1] \
        and point[1] < (rectangle[1] + rectangle[3])