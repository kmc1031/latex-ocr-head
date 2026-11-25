"""TensorFlow setup shared by the segmentation commands."""

import os


def configure_device(name, threads=4):
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    if name == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf

    devices = tf.config.list_physical_devices('GPU')
    if name == 'gpu' and not devices:
        raise RuntimeError('No GPU visible; use the Linux CUDA environment or --device cpu.')
    if devices:
        tf.config.set_visible_devices(devices[:1], 'GPU')
        tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
