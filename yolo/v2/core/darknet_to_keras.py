#!usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import io
import os
from collections import defaultdict

import numpy as np
import requests
from keras import Model
from keras import backend as K
from keras.layers import Input, BatchNormalization, Conv2D, LeakyReLU, Lambda
from keras.layers import concatenate, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
from tqdm import tqdm


def space_to_depth_x2(x):
    """
    Thin wrapper for Tensorflow space_to_depth with block_size=2.
    Import currently required to make Lambda work.
    See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    """
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """
    Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


class DarkNetToKeras(object):
    _URL_WEIGHTS = "http://pjreddie.com/media/files/yolo.weights"
    _URL_CONFIG_FILE = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg"
    _WEIGHTS = "yolov2.weights"
    _CONFIG = "yolov2.cfg"
    _converted_model = "yolov2"
    _directory_model = os.path.join(os.getcwd(), "model_data")

    def __init__(self):
        """
        Converts from Darknet 19 in original format in a format compatible with Keras.
        If the "* .cfg" configuration file does not exist nor the file "* .weights" containing the  weights download
        them from the internet.
        """
        self.output_root = os.path.join(os.getcwd(), self._directory_model)
        self.output_path = os.path.join(os.getcwd(), self._directory_model, self._converted_model)
        self.count = 0
        self.weights_file = None
        self.cfg_parser = None
        self.m_all_layers = []
        self._m_model = None
        if not os.path.exists(os.path.join(os.getcwd(), self._CONFIG)):
            self.config_path = self._download(self._URL_CONFIG_FILE, self._CONFIG)

        else:
            self.config_path = os.path.join(os.getcwd(), self._CONFIG)

        if not os.path.exists(os.path.join(os.getcwd(), self._WEIGHTS)):
            self.weights_path = self._download(self._URL_WEIGHTS, self._WEIGHTS)

        else:
            self.weights_path = os.path.join(os.getcwd(), self._WEIGHTS)

    @staticmethod
    def _download(url, namefile):
        """
        Download the file
        :param url: file to download
        :param namefile: name's file to store on disk
        :return: path
        """
        r = requests.get(url, stream=True)
        total_size = int(r.headers["Content-Length"])
        downloaded = 0  # keep track of size downloaded so far
        chunk_size = 1024
        bars = int(total_size / chunk_size)
        with open(namefile, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=bars, unit="kB", ncols=79,
                              desc=namefile, leave=True):
                f.write(chunk)
                downloaded += chunk_size  # increment the downloaded

        return os.path.expanduser(namefile)

    @staticmethod
    def _unique_config_sections(config_file):
        """
        Convert all config sections to have unique names.
        Adds unique suffixes to config sections for compatibility with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream

    def _load_weights(self):
        # Load weights and config.
        print("==> Loading weights")
        self.weights_file = open(self.weights_path, 'rb')
        weights_header = np.ndarray(
            shape=(4,), dtype='int32', buffer=self.weights_file.read(16))
        print("==> Weights Header: ", weights_header)

    def _parse_configuration(self):
        print("==> Parsing Darknet configuration")
        unique_config_file = self._unique_config_sections(self.config_path)
        self.cfg_parser = configparser.ConfigParser()
        self.cfg_parser.read_file(unique_config_file)

    def _generate_model(self):
        """

        :return:
        """
        print("==> Creating Keras model")
        image_height = int(self.cfg_parser['net_0']['height'])
        image_width = int(self.cfg_parser['net_0']['width'])
        # extract previous layer
        prev_layer = Input(shape=(image_height, image_width, 3))
        all_layers = [prev_layer]

        weight_decay = float(self.cfg_parser['net_0']['decay']
                             ) if 'net_0' in self.cfg_parser.sections() else 5e-4

        for section in self.cfg_parser.sections():
            print("\tParsing section {}".format(section))
            if section.startswith('convolutional'):
                filters = int(self.cfg_parser[section]['filters'])
                size = int(self.cfg_parser[section]['size'])
                stride = int(self.cfg_parser[section]['stride'])
                pad = int(self.cfg_parser[section]['pad'])
                activation = self.cfg_parser[section]['activation']
                batch_normalize = 'batch_normalize' in self.cfg_parser[section]

                # padding='same' is equivalent to Darknet pad=1
                padding = 'same' if pad == 1 else 'valid'

                # Setting weights.
                # Darknet serializes convolutional weights as:
                # [bias/beta, [gamma, mean, variance], conv_weights]
                prev_layer_shape = K.int_shape(prev_layer)

                # TODO: This assumes channel last dim_ordering.
                weights_shape = (size, size, prev_layer_shape[-1], filters)
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)

                print("\tconv2d", "bn" if batch_normalize else "  ", activation, weights_shape)

                conv_bias = np.ndarray(
                    shape=(filters,),
                    dtype='float32',
                    buffer=self.weights_file.read(filters * 4))
                self.count += filters

                if batch_normalize:
                    bn_weights = np.ndarray(
                        shape=(3, filters),
                        dtype='float32',
                        buffer=self.weights_file.read(filters * 12))
                    self.count += 3 * filters

                    # TODO: Keras BatchNormalization mistakenly refers to var
                    # as std.
                    bn_weight_list = [
                        bn_weights[0],  # scale gamma
                        conv_bias,  # shift beta
                        bn_weights[1],  # running mean
                        bn_weights[2]  # running var
                    ]

                conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=self.weights_file.read(weights_size * 4))
                self.count += weights_size

                # DarkNet conv_weights are serialized Caffe-style:
                # (out_dim, in_dim, height, width)
                # We would like to set these to Tensorflow order:
                # (height, width, in_dim, out_dim)
                # TODO: Add check for Theano dim ordering.
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                conv_weights = [conv_weights] if batch_normalize else [
                    conv_weights, conv_bias
                ]

                # Handle activation.
                act_fn = None
                if activation == 'leaky':
                    pass  # Add advanced activation later.
                elif activation != 'linear':
                    raise ValueError(
                        "Unknown activation function `{}` in section {}".format(
                            activation, section))

                # Create Conv2D layer
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)

                if batch_normalize:
                    conv_layer = (BatchNormalization(
                        weights=bn_weight_list))(conv_layer)
                prev_layer = conv_layer

                if activation == 'linear':
                    all_layers.append(prev_layer)
                elif activation == 'leaky':
                    act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)

            elif section.startswith('maxpool'):
                size = int(self.cfg_parser[section]['size'])
                stride = int(self.cfg_parser[section]['stride'])
                self.m_all_layers.append(
                    MaxPooling2D(
                        padding='same',
                        pool_size=(size, size),
                        strides=(stride, stride))(prev_layer))
                prev_layer = all_layers[-1]

            elif section.startswith('avgpool'):
                if self.cfg_parser.items(section) is not []:
                    raise ValueError("{} with params unsupported.".format(section))
                self.m_all_layers.append(GlobalAveragePooling2D()(prev_layer))
                prev_layer = all_layers[-1]

            elif section.startswith('route'):
                ids = [int(i) for i in self.cfg_parser[section]['layers'].split(',')]
                layers = [self.m_all_layers[i] for i in ids]
                if len(layers) > 1:
                    print("\tConcatenating route layers:", layers)
                    concatenate_layer = concatenate(layers)
                    self.m_all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route
                    self.m_all_layers.append(skip_layer)
                    prev_layer = skip_layer

            elif section.startswith('reorg'):
                block_size = int(self.cfg_parser[section]['stride'])
                assert block_size == 2, 'Only reorg with stride 2 supported.'
                self.m_all_layers.append(
                    Lambda(
                        space_to_depth_x2,
                        output_shape=space_to_depth_x2_output_shape,
                        name='space_to_depth_x2')(prev_layer))
                prev_layer = self.m_all_layers[-1]

            elif section.startswith('region'):
                with open('{}_anchors.txt'.format(self.output_path), 'w') as f:
                    print(self.cfg_parser[section]['anchors'], file=f)

            elif (section.startswith('net') or section.startswith('cost') or
                  section.startswith('softmax')):
                pass  # Configs not currently handled during model definition.

            else:
                raise ValueError(
                    'Unsupported section header type: {}'.format(section))

            self.m_all_layers = all_layers

    def _save_model(self):

        # Create and save model.
        model = Model(inputs=self.m_all_layers[0], outputs=self.m_all_layers[-1])
        model.save('{}.h5'.format(self.output_path))
        print("==> Saved Keras model to {}.h5".format(self.output_path))
        # Check to see if all weights have been read.
        remaining_weights = len(self.weights_file.read()) / 4
        print("==> Read {} of {} from Darknet weights.".format(self.count, self.count +
                                                               remaining_weights))
        if remaining_weights > 0:
            print("Warning: {} unused weights".format(remaining_weights))

        plot(model, to_file='{}.png'.format(self.output_path), show_shapes=True)
        print("==> Saved model plot to {}.png".format(self.output_path))
        print(model.summary())
        self._m_model = model

    def extract_model(self):
        self._load_weights()
        self._parse_configuration()
        self._generate_model()
        self._save_model()

    @property
    def model(self):
        return self._m_model

    @model.setter
    def model(self, model):
        self._m_model = model

    def __del__(self):
        if self.weights_file is not None:
            self.weights_file.close()
        # delete model
        del self._m_model
        del self.m_all_layers
