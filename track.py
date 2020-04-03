import numpy as np

from random import randint
from argparse import ArgumentParser
from cv2 import cv2

TEMPLATE_THRESHOLD = 0.6
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

def track(reader, writer, boxes, totalFrames):
    bgsub = cv2.createBackgroundSubtractorKNN()

    # Rewind and train the background subtractor
    trainFrames = 5
    reader.set(cv2.CAP_PROP_POS_FRAMES, int(reader.get(cv2.CAP_PROP_POS_FRAMES)) - trainFrames)

    for _ in range(trainFrames):
        res, colorFrame = reader.read()
        createMask(colorFrame, bgsub, -1)

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

        fgmask = createMask(colorFrame, bgsub, -1)
        boxes = computeBoundingBoxes(fgmask)
        frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

        points, _, _ = cv2.calcOpticalFlowPyrLK(lastFrame, frame, lastPoints, None, **lkParams)
        drawPoints(colorFrame, points)
        drawBoundingBoxes(colorFrame, boxes)

        pos = 0
        for tracker in trackers:
            if tracker.samples is None:
                continue

            ns = len(tracker.samples)
            tracker.update(frame, points[pos:pos+ns], boxes)
            pos += ns

            x, y, w, h = tracker.bounds
            cv2.rectangle(colorFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        lastPoints = np.concatenate(tuple([ t.samples for t in trackers if t.samples is not None ]))
        lastFrame = frame
        writer.write(colorFrame)

def computeBoundingBoxes(mask, minArea=64.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [ cv2.boundingRect(c) for c in contours ]

    # get around a 'bug' or 'feature' of groupRectangles where it removes standalone rectangles
    # (threshold = 0 is a no-op, but threshold = 1 discards rectangles that don't have any sub-rectangles)
    # https://stackoverflow.com/questions/21421070/opencv-grouprectangles-getting-grouped-and-ungrouped-rectangles
    #partitioned, _ = cv2.groupRectangles(rectangles + rectangles, 1, 0.5)
    return rectangles # [ r for r in rectangles if (r[2] * r[3]) >= minArea ]

def createMask(image, bgsub, learn):
    mask = bgsub.apply(image, learn)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    return mask

def drawBoundingBoxes(frame, boxes):
    for rect in boxes:
        cv2.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,255,0), 2)

def rrgb():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

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

class Tracker:
    def __init__(self, bounds, samples):
        self.samples = samples
        self.bounds = bounds

    def update(self, grayscale, samples, newBoxes):
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

        # try and find candidate bounding boxes
        # where we predicted the next box to be
        minX = grayscale.shape[1]
        minY = grayscale.shape[0]
        maxX = 0
        maxY = 0

        for box in newBoxes:
            # Ensure there is overlap in both directions
            if box[0] >= (x + w) or (box[0] + box[2]) <= x:
                continue
            elif box[1] >= (y + h) or (box[1] + box[3]) <= y:
                continue        

            # Compute the intersection
            mx = max(box[0], x)
            my = max(box[1], y)

            mw = min(box[0] + box[2], x + w) - mx
            mh = min(box[1] + box[3], y + h) - my
            
            # threshold the min overlap
            if (((mw * mh) / (box[2] * box[3])) >= 0.5):
                if (mx < minX):
                    minX = mx
                if (my < minY):
                    minY = my
                if ((mx + mw) > maxX):
                    maxX = (mx + mw)
                if ((my + mh) > maxY):
                    maxY = (my + mh)  

        if minX == grayscale.shape[1] or minY == grayscale.shape[0]:
            self.bounds = (x, y, w, h)
        else:
            self.bounds = (minX, minY, maxX - minX, maxY - minY)
            
        print(self.bounds)

        # If any points are now outside the new bounding box, resample the features
        # TODO: Add some sort of threshold as the bounding box size never changes
        # TODO: Worth preserving the in-bounds samples and just resampling the ones that have been lost?
        if not all(isInRectangle(pt, self.bounds) for pt in samples):
            self.samples = sampleFeatures(grayscale, self.bounds)
        else:
            self.samples = samples

        return self.samples