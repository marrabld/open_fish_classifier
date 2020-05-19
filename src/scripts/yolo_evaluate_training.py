import os
import sys
import re

from argparse import ArgumentParser

def resolve_epoch_models(root_dir, start, total):
    pattern = r'^detection_model-ex-0*%d--loss' % start
    models = []

    with os.scandir(os.path.join(root_dir, 'models')) as scanner:
        for entry in scanner:
            if entry.is_file():
                if len(models) > 0 or (len(models) == 0 and re.match(pattern, entry.name)):
                    models.append(entry.path)

                    if len(models) == total:
                        break         

    return models
     
def main(args):
    root_dir = os.path.join('training', args.name)

    if not os.path.isdir(root_dir):
        sys.stderr.write('error: unable to find training run named "%s"\n' % args.name)
        return 1

    # default to evaluating all models
    model_paths = [ os.path.join(root_dir, 'models') ]
    
    if args.start_epoch:
        model_paths = resolve_epoch_models(root_dir, args.start_epoch, args.total_epochs)

        if len(model_paths) == 0:
            sys.stderr.write('error: unable to find a matching model for epoch "%d"\n' % args.start_epoch)
            return 1

    print('info: evaluating models')

    # defer the import until now to avoid loading TensorFlow if there's a 
    # simpler error that's going to break the whole script anyway
    from imageai.Detection.Custom import DetectionModelTrainer
    import tensorflow as tf

    if not tf.test.is_gpu_available:
        print('warning: GPU support not available, evaluating on CPU')

    data_dir = root_dir if args.use_training_dataset else os.path.join(root_dir, 'test')

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=data_dir)

    for model_path in model_paths:
        trainer.evaluateModel(
            model_path=model_path, 
            json_path=os.path.join(root_dir, 'json', 'detection_config.json'), 
            iou_threshold=args.iou_threshold, 
            object_threshold=args.object_threshold, 
            nms_threshold=args.nms_threshold,
        )

if __name__ == '__main__':  
    parser = ArgumentParser('yolo_evaluate_training', 'Evaluate models from a YOLO training run against the test dataset')
    
    parser.add_argument('-i', '--iou-threshold', required=False, type=float, help='IoU threshold to use during evaluation', default=0.5)
    parser.add_argument('-o', '--object-threshold', required=False, type=float, help='Object threshold to use during evaluation', default=0.3)
    parser.add_argument('-n', '--nms-threshold', required=False, type=float, help='Non-maximum suppression threshold to use during evaluation', default=0.5)
    parser.add_argument('-s', '--start-epoch', required=False, type=int, help='A specific epoch to evaluate the model for', default=None)
    parser.add_argument('-t', '--total-epochs', required=False, type=int, help='Total epochs to evaluate (only used when --start-epoch is specified)', default=1)
    parser.add_argument('--use-training-dataset', required=False, action='store_true', help='Run the evaluation against the train/validation datasets instead of the test dataset', default=False)
    parser.add_argument('name', help='Name of the training run to evaluate')

    args = parser.parse_args()
    exit(main(args) or 0)
