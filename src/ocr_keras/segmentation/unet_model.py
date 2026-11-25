"""A compact Keras U-Net with optional center and foreground-to-center heads."""

import keras
from keras import layers
import tensorflow as tf


def build(width, instance):
    def block(inputs, channels, name):
        output = inputs
        for index in (1, 2):
            output = layers.Conv2D(channels, 3, padding='same', use_bias=False,
                                   name=f'{name}_conv{index}')(output)
            output = layers.LayerNormalization(axis=-1, name=f'{name}_norm{index}')(output)
            output = layers.Activation('relu', name=f'{name}_relu{index}')(output)
        return output

    inputs = keras.Input((None, None, 1), name='ink')
    value, skip = inputs, []
    for level in range(3):
        value = block(value, width * 2**level, f'encoder{level}')
        skip.append(value)
        value = layers.MaxPool2D(name=f'pool{level}')(value)
    value = block(value, width * 8, 'bridge')
    for level in reversed(range(3)):
        value = layers.UpSampling2D(interpolation='nearest', name=f'up{level}')(value)
        value = layers.Concatenate(name=f'skip{level}')([value, skip[level]])
        value = block(value, width * 2**level, f'decoder{level}')
    prior = layers.Rescaling(12., offset=-5., name='ink_prior', dtype='float32')(inputs)
    correction = layers.Conv2D(1, 1, name='ink_correction', dtype='float32',
                               kernel_initializer='zeros')(value)
    foreground = layers.Add(name='foreground', dtype='float32')([prior, correction])
    if not instance:
        return keras.Model(inputs, foreground, name='foreground_unet')
    center = layers.Conv2D(1, 1, name='center', dtype='float32',
                           kernel_initializer='zeros',
                           bias_initializer=keras.initializers.Constant(-2.19))(value)
    offset = layers.Conv2D(2, 1, name='offset', dtype='float32', kernel_initializer='zeros')(value)
    outputs = layers.Concatenate(name='instance_maps', dtype='float32')([foreground, center, offset])
    return keras.Model(inputs, outputs, name='instance_unet')


def initialize_instance(model, semantic_model):
    if semantic_model.output_shape[-1] != 1:
        raise ValueError('Instance warm start expects a one-channel foreground model.')
    for layer in semantic_model.layers:
        if layer.weights:
            model.get_layer(layer.name).set_weights(layer.get_weights())


def segmentation_loss(instance):
    def loss(truth, prediction):
        mask, weights = truth[..., 0], truth[..., 1]
        valid = tf.cast(weights > 0, tf.float32)
        foreground = prediction[..., 0]
        probability = tf.sigmoid(foreground)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(mask, foreground)
        bce = tf.reduce_sum(weights * bce) / tf.maximum(1., tf.reduce_sum(weights))
        dice = 1 - (2 * tf.reduce_sum(mask * probability) + 1) / (
            tf.reduce_sum(mask) + tf.reduce_sum(valid * probability) + 1)
        if not instance:
            return bce + dice
        center, center_logits = truth[..., 2], prediction[..., 1]
        peak = tf.cast(center >= 1, tf.float32)
        center_prob = tf.sigmoid(center_logits)
        focal = (peak * (1 - center_prob)**2 * tf.nn.softplus(-center_logits)
                 + (1 - peak) * (1 - center)**4 * center_prob**2 * tf.nn.softplus(center_logits))
        focal = tf.reduce_sum(focal * valid) / tf.maximum(1., tf.reduce_sum(peak))
        delta = tf.abs(truth[..., 3:5] - prediction[..., 2:4])
        huber = tf.where(delta < 1, .5 * delta**2, delta - .5)
        owned = weights * truth[..., 5]
        offset = tf.reduce_sum(huber * owned[..., None]) / tf.maximum(1., 2 * tf.reduce_sum(owned))
        return bce + dice + .5 * focal + offset

    return loss
