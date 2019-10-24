#!usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os

import numpy as np
from PIL import Image, ImageDraw

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

##############################################################################
# Constant path model
##############################################################################
# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.getcwd(), "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(os.getcwd(), "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class LandingZoneConfig(Config):
    """
    Configuration for training on landing zone mate dataset.
    Derives from base Config class and overrides some value.
    """
    # Give the configuration a recognizable name
    NAME = "landing"

    # Adjust for GPU
    IMAGES_PER_GPU = 2

    # Number of class
    NUM_CLASSES = 1 + 2  # background + square mate + circular mate

    # Number of train steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5


class LandingZoneDataset(utils.Dataset):
    def load_landingzone(self, annotations_dir, images_dir) -> None:
        """
        Load COCO like dataset from json.
        Parameters
        ----------
        annotaions_dir : str
            path to the coco annotations json file
        images_dir : str
            the folder path contains images related to the json file
        """
        if os.path.exists(annotations_dir):
            with open(annotations_dir) as infile:
                cocojson = json.load(infile)
        else:
            raise FileNotFoundError(
                "File {} does not exist. Provide a file contains the annotations".format(annotations_dir))

        # add class names using the base method from util.Dataset
        source_name = "landing zone"
        for category in cocojson['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return
            self.add_class(source_name, class_id, class_name)
        # get all annotations
        annotations = {}
        for annotation in cocojson['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # get images and add to the dataset
        seen_images = {}
        for image in cocojson['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(
                        image_id, key))

                image_path = os.path.abspath(
                    os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id) -> tuple:
        """
        Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].

        Parameters
        ----------
        image_id:
            the id of the image to load masks for

        Returns
        -------
        masks:
            A bool array of shape [height, width, instance count] with one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


def train(model) -> None:
    """
    Train the model
    :param model:
    :return:
    """
    # training dataset
    path = "/Users/francesco/Desktop/landingzone/annotations.json"
    images_path = "/Users/francesco/Desktop/landingzone/"
    dataset_train = LandingZoneDataset()
    dataset_train.load_landingzone(path, images_path)
    dataset_train.prepare()

    # validation dataset
    dataset_val = LandingZoneDataset()
    dataset_val.load_landingzone(path, images_path)
    dataset_val.prepare()

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    # # visualize
    # dataset = dataset_train
    # image_ids = np.random.choice(dataset.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #     visualize.display_top_masks(
    #         image, mask, class_ids, dataset.class_names)


if __name__ == '__main__':
    config = LandingZoneConfig()
    config.display()

    # create model in training mode
    model = modellib.MaskRCNN(
        mode="training", config=config, model_dir=MODEL_DIR)
    # init weights
    init_with_weights = "coco"
    if init_with_weights == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with_weights == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif init_with_weights == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    ##############################################################################
    #    Training                                                                #
    ##############################################################################
    #
    # Train in two stages:
    # 1. Only the heads. Here we're freezing all the backbone layers and training
    # only the randomly initialized layers (i.e. the ones that we didn't use
    # pre-trained weights from MS COCO). To train only the head layers, pass
    # layers='heads' to the train() function.
    #
    # 2. Fine-tune all layers. For this simple example it's not necessary, but we're
    #  including it to show the process. Simply pass layers="all to train all layers.
    train(model)
