import os
from pathlib import Path
from argparse import ArgumentParser

def is_valid_path(path):
    return not os.path.splitext(path)[0].endswith('.expected')

def main(args):
    from imageai.Detection.Custom import CustomObjectDetection

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(args.model_path)
    detector.setJsonPath(args.config_path)
    detector.loadModel()

    with os.scandir(args.frame_directory) as scanner:
        entry: object
        for entry in scanner:
            if entry.is_file() and entry.path.endswith('.png'):

                filename, ext = os.path.splitext(os.path.basename(entry.path))

                if not filename.endswith('.expected') and not filename.endswith('.detected'):
                    output_image_path = os.path.join(args.frame_directory, filename + '.detected' + ext)
                    detections = detector.detectObjectsFromImage(entry.path, output_image_path=output_image_path,
                                                    display_object_name=True,
                                                    display_percentage_probability=True,
                                                    minimum_percentage_probability=args.probability)
                    print(f'Detections :: {len(detections)} in {output_image_path}')
                    Path(f'{output_image_path}.' + str(len(detections)) + '.out').touch()


if __name__ == '__main__':
    parser = ArgumentParser('yolo_detect', '')
    parser.add_argument('-m', '--model-path', required=True, help='Path to a trained YOLO model')
    parser.add_argument('-c', '--config-path', required=True, help='Path to the trained YOLO model configuration')
    parser.add_argument('-d', '--frame-directory', required=True, help='Path to the directory containing the test frames')
    parser.add_argument('-p', '--probability', required=False, type=int, default=90)

    args = parser.parse_args()
    exit(main(args) or 0)
