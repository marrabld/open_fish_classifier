from random import randint
from argparse import ArgumentParser
from cv2 import cv2

TEMPLATE_THRESHOLD = 0.6

def track(reader, writer, boxes, totalFrames):
    state = None

    for _ in range(totalFrames):
        res, frame = reader.read()

        if not res:
            print('ran out of frames')
            break

        # initialise the state on the first frame
        if state is None:
            state = [ (i, frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], b, __randrgb()) for i, b in enumerate(boxes) ]
        else:  
            # compute the next state from the frame and the current state
            state = __nextState(frame, state)
            __writeTrackerState(writer, frame, state)

def __randrgb():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def __writeTrackerState(writer, frame, current):
    for state in current:
        id, _, box, color = state
        cv2.rectangle(frame, box[0:2], (box[0] + box[2], box[1] + box[3]), color, 3)
        cv2.putText(frame, str(id), box[0:2], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    writer.write(frame)

def __nextState(frame, current):
    nextState = []

    for state in current:
        id, template, lastBox, color = state
        _, maxVal, _, maxLoc = cv2.minMaxLoc(cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED))

        if maxVal >= TEMPLATE_THRESHOLD:
            nextBox = (maxLoc[0], maxLoc[1], lastBox[2], lastBox[3])
            nextTemplate = frame[nextBox[1]:nextBox[1]+nextBox[3], nextBox[0]:nextBox[0]+nextBox[2]].copy()
            nextState.append((id, nextTemplate, nextBox, color))

    return nextState