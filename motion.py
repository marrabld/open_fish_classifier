import numpy as np
import json

from itertools import chain
from random import randint
from cv2 import cv2

def shapeToBox(shape):
    return (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))

def randpt(box):
    return [ float(randint(box[0], box[0] + box[2] - 1)), float(box[1] + (box[3] / 2)) ]

def sampleBox(box):
    return [ randpt(box) for i in range(4) ]

def drawPoints(frame, points):
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

def sampleFish(grayscale, boundingBox, count, sigma=0.33):
    # Take a crop of the image where the bounding box is located,
    # then attempt to find edges using the Canny
    crop = grayscale[boundingBox[1]:boundingBox[1]+boundingBox[3], boundingBox[0]:boundingBox[0]+boundingBox[2]]
    mean = np.median(crop)
    canny = cv2.Canny(crop, mean * (1 - sigma), mean * (1 + sigma))
    filtered = np.where(canny > 0)

    edgeCount = len(filtered[0])
    samples = [ None ] * min(count, edgeCount)

    for i in range(len(samples)):
        index = randint(0, edgeCount - 1)
        samples[i] = [ boundingBox[0]+filtered[1][index], boundingBox[1]+filtered[0][index] ]

    return samples
    

video = cv2.VideoCapture(r"E:\Work\data\fish\transfer_1908888_files_fdbb9eb7\A000001_R.avi")

# seed the background subtraction algorithm
bgsub = cv2.createBackgroundSubtractorKNN()
video.set(cv2.CAP_PROP_POS_FRAMES, 4550)

for i in range(1):
    _, seedFrame = video.read()
    bgsub.apply(seedFrame, 1)

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

lastPoints =  np.array(list(chain.from_iterable([ sampleFish(frame, b, 4) for b in boxes ])), dtype=np.float32)
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
    
    if lastFrame is not None:
        points, status, error = cv2.calcOpticalFlowPyrLK(lastFrame, frame, lastPoints, None, **lkParams)
        drawPoints(colorFrame, points)
        lastPoints = points
        #print(status)
        #cv2.imshow('frame', colorFrame)
        writer.write(colorFrame)
        
#
    key = cv2.waitKey()
    lastFrame = frame

video.release()
cv2.destroyAllWindows()