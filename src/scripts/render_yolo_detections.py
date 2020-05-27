import cv2
import numpy as np
import os
import json

from argparse import ArgumentParser

# hardcoded input image size to use with YOLO (hardcoded in ImageAI)
INPUT_SIZE = 416

def compute_predictions(raw_image, model, anchors, utils, threshold):
    height, width, _ = raw_image.shape

    image = cv2.resize(raw_image, (INPUT_SIZE, INPUT_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.
    image = np.expand_dims(image, 0)

    predictions = model.predict(image)
    boxes = []

    for idx, [ prediction ] in enumerate(predictions):
        boxes += utils.decode_netout(prediction, anchors[idx], threshold, INPUT_SIZE, INPUT_SIZE)

    utils.correct_yolo_boxes(boxes, height, width, INPUT_SIZE, INPUT_SIZE)
    return boxes

def nms(predictions, threshold=0.5):
    # transform the predictions into a numpy array
    boxes = np.empty((len(predictions), 5), dtype='float')

    for i, p in enumerate(predictions):
        boxes[i][0:4] = (p.xmin, p.ymin, p.xmax, p.ymax)
        boxes[i][4] = np.max(p.classes)

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

def get_top_class_index(prediction, labels):
    ranked = np.argsort(prediction.classes)

    if labels[ranked[-1]] == 'fish' and prediction.classes[ranked[-2]] > 0.5:
        return ranked[-2]

    return ranked[-1]

def get_class_color(label_index):
    color_wheel = [ 
        (0xFF, 0xFF, 0xFF), 
        (0x77, 0x66, 0xCC),
        (0x77, 0xCC, 0xDD),
        (0x33, 0x77, 0x11),
        (0x00, 0x00, 0xFF),
        (0x99, 0x44, 0xAA),
        (0x00, 0xFF, 0x00),
        (0x33, 0x99, 0x99),
        (0x55, 0x22, 0x88),
        (0x00, 0x11, 0x66),
        (0x00, 0xFF, 0xFF),
        (0x88, 0x88, 0x88)
    ]

    return color_wheel[label_index % len(color_wheel)]

def main(args):
    with open(args.config_path) as config_file:
        config = json.load(config_file)
        labels = config['labels']
        anchors = config['anchors']

    reader = cv2.VideoCapture(args.video_path)
    filename, ext = os.path.splitext(os.path.basename(args.video_path))

    # Determine the output path and ensure all intermediate directories exist
    output_path = os.path.abspath(args.output_path or os.path.join(os.path.dirname(args.video_path), filename + '.detections' + ext))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not reader.isOpened():
        print('error: unable to open "%s"; please ensure it exists' % args.video_path)
        return 1

    frame_size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), reader.get(cv2.CAP_PROP_FPS), frame_size)

    frame_num = 0

    from imageai.Detection.YOLOv3.models import yolo_main
    from imageai.Detection.Custom import CustomDetectionUtils, CustomObjectDetection
    from keras import Input

    utils = CustomDetectionUtils(labels)
    model = yolo_main(Input(shape=(None, None, 3)), 3, len(labels))
    model.load_weights(args.model_path)

    # re-usable colors
    white = (255, 255, 255)
    black = (0, 0, 0)

    while reader.isOpened():
        res, frame = reader.read()

        if not res:
            break

        frame_num += 1
        print('info: processing frame %d' % frame_num)

        predictions = compute_predictions(frame, model, anchors, utils, args.object_threshold)
        predictions = nms(predictions, args.nms_threshold)
        totals = dict()

        for prediction in predictions:
            cls_index = get_top_class_index(prediction, labels)
            color = get_class_color(cls_index)
            label = labels[cls_index]

            totals[label] = 1 if not label in totals else totals[label] + 1
            
            cv2.rectangle(frame, (prediction.xmin, prediction.ymin), (prediction.xmax, prediction.ymax), color, 1)
            cv2.putText(frame, '%s: %.3f' % (label, prediction.classes[cls_index]), (prediction.xmin, prediction.ymin - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, white, 1)

        # draw totals
        text_pos = (15, frame_size[1] - 30)

        cv2.putText(frame, 'total: %d' % len(predictions), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, 4)
        cv2.putText(frame, 'total: %d' % len(predictions), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

        for item in sorted(totals.items(), reverse=True, key=lambda x: x[1]):
            text_pos = (text_pos[0], text_pos[1] - 30)
            cv2.putText(frame, '%s: %d' % item, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, 4)
            cv2.putText(frame, '%s: %d' % item, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

        writer.write(frame)

    writer.release()
    reader.release()

if __name__ == '__main__':
    parser = ArgumentParser('yolo_video')
    parser.add_argument('-m', '--model-path', required=True, help='Path to the pre-trained YOLOv3 model to use for detections')
    parser.add_argument('-c', '--config-path', required=True, help='Path to the configuration file for this model (typically detection_config.json)')
    parser.add_argument('-o', '--output-path', help='Path to output the video to (defaults to the same directory as the input video)', default=None)
    parser.add_argument('--object-threshold', required=False, type=float, help='Minimum "object-ness" required to consider a YOLO detection valid', default=0.5)
    parser.add_argument('--nms-threshold', required=False, type=float, help='Non-Maximum Supression threshold used to de-duplicate similar detections', default=0.4)
    parser.add_argument('video_path', help='Path to video file to run the detections on')

    args = parser.parse_args()
    exit(main(args) or 0)