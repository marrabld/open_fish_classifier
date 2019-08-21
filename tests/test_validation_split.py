import sys
import os

sys.path.append(os.path.abspath('..'))
import src.tools.pre_processing_tools as ppt

yolo = ppt.YoloTools()

yolo.split_training_and_validation('../data/labels/fish')