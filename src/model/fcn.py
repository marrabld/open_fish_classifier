import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from src.tools.meta import log as log
import os


def fcn_model(len_classes=5, dropout_rate=0.2, multi_gpu=False):
    # Input layer
    input = tf.keras.layers.Input(shape=(None, None, 3))

    # A convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # A convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

        # A convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

        # A convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=len_classes, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    predictions = tf.keras.layers.Activation('softmax')(x)

    if multi_gpu:
        tf.compat.v1.disable_eager_execution()
        log.info('Attempting multi GPU')

        _model = tf.keras.Model(inputs=input, outputs=predictions)
        model = multi_gpu_model(_model, gpus=4)

    else:
        model = tf.keras.Model(inputs=input, outputs=predictions)
    print(model.summary())

    return model


def train(model, train_generator, val_generator, epochs=50):
    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path,
                              'model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_loss_{'
                              'val_loss:.2f}_val_acc_{val_accuracy:.2f}.h5')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                                      save_best_only=True, verbose=1)],
                        validation_data=val_generator,
                        validation_steps=len(val_generator))

    return history
