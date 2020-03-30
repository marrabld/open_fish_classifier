import numpy as np
import json

from itertools import chain
from random import randint
from cv2 import cv2
from scipy.spatial.distance import cdist

def shapeToBox(shape):
    return (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))

def randpt(box):
    return [ float(randint(box[0], box[0] + box[2] - 1)), float(box[1] + (box[3] / 2)) ]

def sampleBox(box):
    return [ randpt(box) for i in range(4) ]

def drawPoints(frame, points):
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

def computeDistances(pts1, pts2):
    assert(len(pts1) == len(pts2))
    return [ np.linalg.norm(pts2[i] - pts1[i]) for i in range(len(pts1)) ]
    
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


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

video = cv2.VideoCapture(r"E:\Work\data\fish\transfer_1908888_files_fdbb9eb7\A000001_R.avi")

# seed the background subtraction algorithm
bgsub = cv2.createBackgroundSubtractorMOG2()
video.set(cv2.CAP_PROP_POS_FRAMES, 4875)

_, colorSeedFrame = video.read()
bgsub.apply(colorSeedFrame)
seedFrame = cv2.cvtColor(colorSeedFrame, cv2.COLOR_BGR2GRAY)

boxes = None

with open('./regions.json', 'r') as f:
    regions = json.load(f)
    boxes = [ shapeToBox(r['shape_attributes']) for r in regions ]

# Tweak-able LK parameters
lkParams = dict(
    winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

startFrame = 45631
video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

_, colorFrame = video.read()
frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)
trackers = [ Tracker(b, sampleFeatures(frame, b)) for b in boxes ]
lastPoints = np.concatenate(tuple([ t.samples for t in trackers ]))

#lastPoints =  np.array(list(chain.from_iterable([ sampleFish(frame, b, 4) for b in boxes ])), dtype=np.float32)
lastFrame = frame
writer = cv2.VideoWriter('./motion.avi', cv2.VideoWriter_fourcc(*'XVID'), video.get(cv2.CAP_PROP_FPS), (1920, 1080))

#for box in boxes:
#    crop = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].copy()
#    colorCrop = colorFrame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
#    mean = np.mean(crop)
#    canny = cv2.Canny(crop, mean * (1 - 0.33) , mean * (1 + 0.33))
#    print(cannt)
#    contours, _ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cv2.imshow('image', crop)
#    cv2.waitKey()
#    cv2.imshow('image', canny)
#    cv2.waitKey()
#    cv2.drawContours(crop, contours, -1, (0,0,255), 2)
#    cv2.imshow('image', crop)
#    cv2.waitKey()


for i in range(200):
    _, colorFrame = video.read()
    frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

    #mask = cv2.absdiff(seedFrame, frame) # bgsub.apply(colorFrame)
    ##_, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
#
    #cv2.imshow('diff', mask)
    #cv2.waitKey()
    
    if lastFrame is not None:
        points, status, error = cv2.calcOpticalFlowPyrLK(lastFrame, frame, lastPoints, None, **lkParams)
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

        #print(status)
        #cv2.imshow('frame', colorFrame)
        lastPoints = np.concatenate(tuple([ t.samples for t in trackers if t.samples is not None ]))
        writer.write(colorFrame)
        cv2.imshow('frame', colorFrame)
        cv2.waitKey()
        
    lastFrame = frame

video.release()
cv2.destroyAllWindows()