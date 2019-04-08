#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.layers import Lambda
from keras.layers.merge import concatenate

from .darknet19 import compose, Darknet19, _darknet_Conv2D_BN_Leaky, _darknet_Conv2D
from .darknet_to_keras import DarkNetToKeras


def space_to_depth_x2(x):
    """
    Thin wrapper for Tensorflow space_to_depth with block_size=2.
    Import currently required to make Lambda work.
    See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    """
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """
    Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0],
            input_shape[1] // 2,
            input_shape[2] // 2,
            4 * input_shape[3]) if input_shape[1] else (input_shape[0], None, None, 4 * input_shape[3])


voc_anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class Yolo_V2:
    _model = None

    def __init__(self):
        original = DarkNetToKeras()
        original.extract_model()
        self.model = original.model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @classmethod
    def _body(cls, inputs, num_anchors, num_classes):
        """
        Create YOLO_V2 model CNN body in Keras.
        """
        darknet = Model(inputs, Darknet19(inputs).darknet.input)
        conv20 = compose(
            _darknet_Conv2D_BN_Leaky(1024, (3, 3)),
            _darknet_Conv2D_BN_Leaky(1024, (3, 3)))(darknet.output)

        conv13 = darknet.layers[43].output
        conv21 = _darknet_Conv2D_BN_Leaky(64, (1, 1))(conv13)
        #
        conv21_reshaped = Lambda(
            space_to_depth_x2,
            output_shape=space_to_depth_x2_output_shape,
            name='space_to_depth')(conv21)

        x = concatenate([conv21_reshaped, conv20])
        x = _darknet_Conv2D_BN_Leaky(1024, (3, 3))(x)
        x = _darknet_Conv2D(num_anchors * (num_classes + 5), (1, 1))(x)
        return Model(inputs, x)

    @classmethod
    def head(cls, feats, anchors, num_classes):
        """
        Convert final layer features to bounding box parameters.
        Parameters
        ----------
        :param feats: [tensor]
        :param anchors : [array-like] Anchor box widths and heights.
        :param num_classes : [int] Number of target classes.

        Returns
        -------
        :param box_xy: [tensor] x, y box predictions adjusted by spatial location in conv layer.
        :param box_wh: [tensor] w, h box predictions adjusted by anchors and conv spatial resolution.
        :param box_conf: [tensor] Probability estimate for whether each box contains any object.
        :param box_class_pred: [tensor] Probability distribution estimate for each box over class labels.
        """

        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
        # Dynamic implementation of conv dims for fully convolutional model.
        conv_dims = K.shape(feats)[1:3]  # assuming channels last
        # In YOLO the height index is the inner most iteration.
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
        #
        conv_width_index = K.tile(
            K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        feats = K.reshape(
            feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        box_xy = K.sigmoid(feats[..., :2])
        box_wh = K.exp(feats[..., 2:4])
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.softmax(feats[..., 5:])

        # Adjust predictions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        return box_xy, box_wh, box_confidence, box_class_probs

    def __del__(self):
        del self._model


def _filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold
    #
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def _boxes_to_corners(box_xy, box_wh):
    """
    Convert YOLO box predictions to bounding box corners.
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def evaluate(outputs, image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Evaluate YOLO model on given input batch and return filtered boxes.
    :param outputs: [tuple] contains return value from method head (box_xy, box_wh, box_confidence, box_class_probs)
    :param image_shape: [tensor] image'tensor
    :param max_boxes: [int] number of max boxes generated
    :param score_threshold: (optional)
    :param iou_threshold: intersection threshold value (optional)
    :return boxes:
    :return scores:
    :return classes:
    """
    box_xy, box_wh, box_confidence, box_class_probs = outputs
    boxes = _boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = _filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)
    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    #
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes


def YoloV2(anchors, num_classes):
    yolo = Yolo_V2()
    outputs = yolo.head(feats=yolo.model.output, anchors=anchors, num_classes=num_classes)
    return yolo.model, outputs
