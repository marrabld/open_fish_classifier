from argparse import ArgumentParser
from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
import pylab as plt
import os
import pathlib

def main(args):
    """
    Train on the crop dataset.

    :return:
    """
    BATCH_SIZE = 32
    IMG_HEIGHT = 160  # 224
    IMG_WIDTH = 160  # 224

    SHUFFLE_BUFFER_SIZE = 1000
    IMG_SIZE = 160  # All images will be resized to 160x160
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    data_dir = pathlib.Path(args.input)
    image_count = len(list(data_dir.glob('*/*.jpg')))

    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        # return tuple(ds)
        return ds

    train_ds = prepare_for_training(labeled_ds)

    image_batch, label_batch = next(iter(train_ds))

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # from keras.applications.inception_v3 import InceptionV3
    # load model
    # base_model = InceptionV3()
    feature_batch = base_model(image_batch)
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(5)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # summarize the model
    model.summary()

    len(model.trainable_variables)

    initial_epochs = 10
    validation_steps = 20


if __name__ == '__main__':
    parser = ArgumentParser('gen_frame_annotations',
                            description='Train on the crop dataset')
    parser.add_argument('--input', help='input directory containing crop images', required=True)


    exit(main(parser.parse_args()) or 0)
