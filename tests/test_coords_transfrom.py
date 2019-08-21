import sys
import os

sys.path.append(os.path.abspath('..'))
import src.tools.pre_processing_tools as ppt
import skimage.io as io
import os
from src.tools.helper_functions import timeit

import manage

yolo = ppt.YoloTools()


yolo.generate_yolo_labels_from_bbtool('C:/Users/radae/Desktop/Machine Learning/labels',
                                      'C:/Users/radae/Desktop/Machine Learning/labels/fish', 0)
