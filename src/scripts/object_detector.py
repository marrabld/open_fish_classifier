import json
import numpy as np
import cv2

from imageai.Detection.YOLOv3.models import yolo_main
from imageai.Detection.Custom import CustomDetectionUtils # TODO: replace this with custom code
from keras import Input

class ObjectDetector:
    def __init__(self, model, labels, anchors):
        self.model = model
        self.labels = labels
        self.anchors = anchors

        self._utils = CustomDetectionUtils(labels)
        self._input_size = (416, 416) # hardcoded by ImageAI

    @staticmethod
    def from_config(weights_path, config_path):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            labels = config['labels']
            anchors = config['anchors']

        model = yolo_main(Input(shape=(None, None, 3)), 3, len(labels))
        model.load_weights(weights_path)

        return ObjectDetector(model, labels, anchors)

    def detect(self, image, object_threshold, nms_threshold):
        height, width, _ = image.shape

        image = cv2.resize(image, self._input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.
        image = np.expand_dims(image, 0)

        predictions = self.model.predict(image)
        boxes = []

        for idx, [ prediction ] in enumerate(predictions):
            boxes += self._utils.decode_netout(prediction, self.anchors[idx], object_threshold, self._input_size[0], self._input_size[1])

        self._utils.correct_yolo_boxes(boxes, height, width, self._input_size[0], self._input_size[1])
        return _non_maximum_suppression(boxes, nms_threshold)

def _non_maximum_suppression(predictions, threshold):
    # transform the predictions into a numpy array
    boxes = np.empty((len(predictions), 5), dtype='float')

    for i, p in enumerate(predictions):
        boxes[i,:4] = (p.xmin, p.ymin, p.xmax, p.ymax)
        boxes[i,4] = np.max(p.classes)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    cp = boxes[:,4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by highest classification probability (cp)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(cp)

    # keep track of the indices we choose
    pick = []

    while len(indices) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indices
        i = indices[-1]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[indices[:-1]])
        yy1 = np.maximum(y1[i], y1[indices[:-1]])
        xx2 = np.minimum(x2[i], x2[indices[:-1]])
        yy2 = np.minimum(y2[i], y2[indices[:-1]])

        # compute the width/height of the intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the IoUs
        intersection = w * h
        union = area[indices[:-1]] - intersection + area[i]
        iou = intersection / union

        # delete all indexes from the index list that have IoU > threshold
        indices = np.delete(indices, np.concatenate(([len(indices) - 1], np.where(iou > threshold)[0])))

    return [ predictions[i] for i in pick ]
