import os

from argparse import ArgumentParser

def is_valid_path(path):
    return not os.path.splitext(path)[0].endswith('.expected')

def main(args):
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=args.directory)

    trainer.evaluateModel(
        model_path=args.model_path, 
        json_path=args.config_path, 
        iou_threshold=args.iou_threshold, 
        object_threshold=args.object_threshold, 
        nms_threshold=args.nms_threshold
    )

if __name__ == '__main__':
    parser = ArgumentParser('yolo_evaluate_training', 'Evaluate a training run against a specific model')
    parser.add_argument('-m', '--model-path', required=True, help='Path to a trained YOLO model')
    parser.add_argument('-c', '--config-path', required=True, help='Path to the trained YOLO model configuration')
    parser.add_argument('-i', '--iou-threshold', required=False, type=float, help='IoU threshold to use during evaluation', default=0.5)
    parser.add_argument('-o', '--object-threshold', required=False, type=float, help='Object threshold to use during evaluation', default=0.3)
    parser.add_argument('-n', '--nms-threshold', required=False, type=float, help='Non-maximum suppression threshold to use during evaluation', default=0.5)
    parser.add_argument('-d', '--directory', required=True, help='Path to the root directory containing the test dataset')

    args = parser.parse_args()
    exit(main(args) or 0)

