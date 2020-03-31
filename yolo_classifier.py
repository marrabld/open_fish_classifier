from src.model import yolov3
import os
import tensorflow as tf
from os.path import join as join
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from generator import Generator
from src.tools.meta import log as log

import pickle

DATA_DIR = '/home/danm/data/fishml'


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Only need to call this the first time.  After that we can just load the keras model
yolov3.build_and_save_model(os.path.join(DATA_DIR, 'model_weights', 'yolov3.weights'))


