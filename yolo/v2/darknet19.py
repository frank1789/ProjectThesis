#!usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
from functools import partial

from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2

from .util import DarkNetYoloCommonLayer, compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


class DarkNet19(DarkNetYoloCommonLayer):
    """Generate Darknet-19 model for Imagenet classification."""

    def __init__(self, inputs):
        body = self.darknet_body()(inputs)
        logitstic = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
        self._darknet = Model(inputs, logitstic)

    @classmethod
    def darknet_body(cls):
        """Generate first 18 conv layers of Darknet-19."""
        return compose(
            cls.DarknetConv2D_BN_Leaky(32, (3, 3)),
            MaxPooling2D(),
            cls.DarknetConv2D_BN_Leaky(64, (3, 3)),
            MaxPooling2D(),
            cls.bottleneck_block(128, 64),
            MaxPooling2D(),
            cls.bottleneck_block(256, 128),
            MaxPooling2D(),
            cls.bottleneck_x2_block(512, 256),
            MaxPooling2D(),
            cls.bottleneck_x2_block(1024, 512))

    @property
    def darknet(self):
        return self._darknet


if __name__ == '__main__':
    from keras.layers import Input

    input = Input(shape=(690, 690, 3))
    model = DarkNet19(input)
    model.darknet.summary()
