import cv2
import pickle

from argparse import ArgumentParser

def annotate(video_path, detector, stride):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // stride
    frames = [ None ] * total_frames

    for frame_index in range(total_frames):
        frame_number = frame_index * stride
        
        if stride > 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        res, frame = video.read()

        if not res:
            print('warning: ran out of frames prematurely')
            break

        print('info: processed up to frame %d' % frame_index)

        _, detected_objects = detector.detectObjectsFromImage(
            input_image=frame,
            input_type='array',
            output_type='array',
            minimum_percentage_probability=30,
            display_percentage_probability=False,
            display_object_name=False
        )

        frames[frame_index] = [ [ frame_number, o['name'], o['percentage_probability'] / 100, *o['box_points'] ] for o in detected_objects ]

    return frames

def main(args):
    from imageai.Detection.Custom import CustomObjectDetection

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(args.model_path)
    detector.setJsonPath(args.config_path)
    detector.loadModel()

    frames = annotate(args.video_path, detector, args.stride)
    
    with open(args.output_path, 'wb') as of:
        pickle.dump(frames, of)

if __name__ == '__main__':
    parser = ArgumentParser('maxn')
    parser.add_argument('-m', '--model-path', required=True, help='Path to model')
    parser.add_argument('-c', '--config-path', required=True)
    parser.add_argument('-s', '--stride', type=int, required=False, default=1)
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('video_path', help='Path to input video file')

    args = parser.parse_args()
    exit(main(args) or 0)

