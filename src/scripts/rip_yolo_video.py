import os
import cv2
from argparse import ArgumentParser
from collections import Counter
def main(args):
    from imageai.Detection.Custom import CustomObjectDetection
    import tensorflow as tf
    print('is gpu available', tf.test.is_gpu_available())
    reader = cv2.VideoCapture(args.video_path)
    filename, ext = os.path.splitext(os.path.basename(args.video_path))
    # Determine the output path and ensure all intermediate directories exist
    if args.output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.video_path)), filename + '.detections' + ext)
    else:
        output_path = os.path.abspath(args.output_path)
    print(f'Saving file {output_path}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not reader.isOpened():
        print('error: unable to open "%s"; please ensure it exists' % args.video_path)
        return 1
    frame_size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), reader.get(cv2.CAP_PROP_FPS), frame_size)
    # writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), reader.get(cv2.CAP_PROP_FPS), frame_size)

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(args.model_path)
    detector.setJsonPath(args.config_path)
    detector.loadModel()
    frame_num = 0
    while reader.isOpened():
        res, frame = reader.read()
        if not res:
            break
        frame_num += 1
        print('info: processing frame %d' % frame_num)
        detected_frame, detections = detector.detectObjectsFromImage(
            input_image=frame,
            input_type='array',
            output_type='array',
            minimum_percentage_probability=args.probability,
            display_object_name=True,
            display_percentage_probability=True
        )
        totals = Counter((d['name'] for d in detections))
        text_pos = (15, frame_size[1] - 30)
        white = (255, 255, 255)
        black = (0, 0, 0)
        cv2.putText(detected_frame, 'total: %d' % len(detections), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, black, 4)
        cv2.putText(detected_frame, 'total: %d' % len(detections), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, white, 2)
        for item in sorted(totals.items(), reverse=True, key=lambda x: x[1]):
            text_pos = (text_pos[0], text_pos[1] - 30)
            cv2.putText(detected_frame, '%s: %d' % item, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, black, 4)
            cv2.putText(detected_frame, '%s: %d' % item, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, white, 2)
        writer.write(detected_frame)
    writer.release()
    reader.release()
if __name__ == '__main__':
    parser = ArgumentParser('yolo_video')
    parser.add_argument('-m', '--model-path', help='Path to the pre-trained YOLOv3 model to use for detections')
    parser.add_argument('-c', '--config-path', help='Path to the configuration file for this model (typically detection_config.json)')
    parser.add_argument('-o', '--output-path', required=False, help='Path to output the video to (defaults to the same directory as the input video)', default=None)
    parser.add_argument('video_path', help='Path to video file to run the detections on')
    parser.add_argument('-p', '--probability', required=False, type=int, default=90)
    args = parser.parse_args()
    exit(main(args) or 0)