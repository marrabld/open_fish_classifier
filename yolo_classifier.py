from src.model import yolov3
import tensorflow as tf
from os.path import join as join
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from generator import Generator
from src.tools.meta import log as log

import pickle

DATA_DIR = '/home/danm/data/fishml/output'


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = yolov3.yolov3_model()

    weight_reader = yolov3.WeightReader('yolov3.weights')
    # set the model weights into the model
    weight_reader.load_weights(model)
    # save the model to file
    model.save('model.h5')

    # train_datagen = Generator(join(DATA_DIR, 'train'), BATCH_SIZE=16, shuffle_images=True)
    # val_datagen = Generator(join(DATA_DIR, 'val'), BATCH_SIZE=16, shuffle_images=True)
    #
    # history = fcn.train(model, train_datagen, val_generator=train_datagen, epochs=500)

# save_obj(history, 'fcn_history')
