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

    # TODO: Make these configurable in the ctor
    featureParams = dict(
        maxCorners = 10,
        qualityLevel = 0.02,
        minDistance = 0.02 * max(w, h)
    )

    crop = grayscale[y:y+h, x:x+w]
    corners = cv2.goodFeaturesToTrack(crop, **featureParams)

    # Normalise the points back to the parent image
    return np.array([ [ x + c[0][0], y + c[0][1] ] for c in corners ], dtype=np.float32)

class Tracker:
    def __init__(self, bounds, samples):
        self.samples = samples
        self.bounds = bounds

        x, y, w, h = bounds
        sampleCenter = np.mean(self.samples, axis=0)
        boundsCenter = [ x + (w / 2), y + (h / 2) ]

        self.offset = np.array([ sampleCenter[0] - boundsCenter[0], sampleCenter[1] - boundsCenter[1]], dtype=np.float32)

    def update(self, grayscale, samples, statuses):
        sampleCenter = np.mean(samples, axis=0)
        boundsCenter = sampleCenter - self.offset

        self.bounds = (
            boundsCenter[0] - (self.bounds[2] / 2),
            boundsCenter[1] - (self.bounds[3] / 2),
            self.bounds[2],
            self.bounds[3]
        )

def sampleFish(grayscale, boundingBox, count, sigma=0.33):
    # Take a crop of the image where the bounding box is located,
    # then attempt to find edges using the Canny algorithm
    x, y, w, h = boundingBox
    crop = grayscale[y:y+h, x:x+w]
    mean = np.mean(crop) # TODO: Compare median vs mean results

    canny = cv2.Canny(crop, mean * (1 - sigma), mean * (1 + sigma))
    filtered = np.where(canny > 0)

    edgeCount = len(filtered[0])

    if (edgeCount < count):
        return []

    samples = [ 
        [boundingBox[0] + filtered[1][0], boundingBox[1] + filtered[0][0]],
        [boundingBox[0] + filtered[1][-1], boundingBox[1] + filtered[0][-1]] 
    ]

    for i in range(2, count):
        index = randint(0, edgeCount - 1)
        samples.append([ boundingBox[0]+filtered[1][index], boundingBox[1]+filtered[0][index] ])

    return samples


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

video = cv2.VideoCapture(r"C:\Users\254288b\development\data\fish\transfer_1908888_files_fdbb9eb7\A000001_R.avi")

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
lastFrame = None
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
        lastPoints = points

        pos = 0
        for tracker in trackers:
            ns = len(tracker.samples)
            tracker.update(frame, points[pos:pos+ns], status[pos:pos+ns])
            pos += ns

            x, y, w, h = tracker.bounds
            print((x, y, w, h))
            cv2.rectangle(colorFrame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

        #print(status)
        #cv2.imshow('frame', colorFrame)
        writer.write(colorFrame)
        cv2.imshow('frame', colorFrame)
        cv2.waitKey()
        
    lastFrame = frame

video.release()
cv2.destroyAllWindows()