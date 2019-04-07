#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""Miscellaneous utility functions."""

import functools
from functools import reduce, partial

import tensorflow as tf
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def space_to_depth_x2(x):
    """
    Thin wrapper for Tensorflow space_to_depth with block_size=2.
    """
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """
    Wrapper to set Darknet weight regularizer for Convolution2D.
    """
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


class DarkNetYoloCommonLayer:

    @staticmethod
    def DarknetConv2D_BN_Leaky(*args, **kwargs):
        """
        Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
        """
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        return compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            BatchNormalization(),
            LeakyReLU(alpha=0.1))

    @staticmethod
    def bottleneck_block(outer_filters, bottleneck_filters):
        """
        Bottleneck block of 3x3, 1x1, 3x3 convolutions.
        """
        return compose(
            DarkNetYoloCommonLayer.DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
            DarkNetYoloCommonLayer.DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
            DarkNetYoloCommonLayer.DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))

    @staticmethod
    def bottleneck_x2_block(outer_filters, bottleneck_filters):
        """
        Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions.
        """
        return compose(
            DarkNetYoloCommonLayer.bottleneck_block(outer_filters, bottleneck_filters),
            DarkNetYoloCommonLayer.DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
            DarkNetYoloCommonLayer.DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))
