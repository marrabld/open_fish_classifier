import tensorflow as tf

if not tf.test.is_gpu_available():
    print('warning', 'GPU support is not available, training on CPU')