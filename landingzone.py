#!usr/bin/env python3
# -*- coding: utf-8 -*-

import signal
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from PIL import Image, ImageDraw
import numpy as np
import sys
import os
import json
import warnings
from staticsanalysis import HistoryAnalysis

warnings.filterwarnings('ignore', category=FutureWarning)


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# check if travis environment
is_travis = 'TRAVIS' in os.environ


def handler(signum, frame):
    print("Times up! Exiting...")
    exit(0)


# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

##############################################################################
# Constant path model
##############################################################################
# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.getcwd(), "logs")

# Local path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(os.getcwd(), "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), "logs")


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
    NUM_CLASSES = 1 + 1  # background + landing mate

    # Number of train steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 25


class LandingZoneDataset(utils.Dataset):
    def load_landingzone(self, annotations_dir, images_dir) -> None:
        """
        Load COCO like dataset from json.
        Parameters
        ----------
        annotations_dir : str
            path to the coco annotations json file
        images_dir : str
            the folder path contains images related to the json file
        """
        self.add_class("landing", 1, "landing")
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


def train(args, model) -> None:
    """
    Train the model
    :param model:
    :return:
    """
    # training dataset
    path = args.annotations
    images_path = args.dataset
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
    history = model.train(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=200,
                          layers='heads')
    HistoryAnalysis.plot_history(history, "landzone_head")

    print("Train all layers")
    hist = model.train(dataset_train, dataset_val,
                       learning_rate=config.LEARNING_RATE,
                       epochs=200,
                       layers='all')
    HistoryAnalysis.plot_history(hist, "landzone_all_layer")

    # # visualize
    # dataset = dataset_train
    # image_ids = np.random.choice(dataset.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #     visualize.display_top_masks(
    #         image, mask, class_ids, dataset.class_names)


def init_weights(args, model):
    # weights_path = None
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    return model


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect landing zone mate.')
    parser.add_argument('-a', '--annotations', required=True,
                        metavar="/path/to/dataset/annotations.json",
                        help='Path to annotations json file')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # check existence of travis then install signal handler
    if is_travis:
        print("work on travis: ", is_travis)
        signal.signal(signal.SIGALRM, handler)
        # Set alarm for 5 minutes
        signal.alarm(300)

    # initialize configuration
    config = LandingZoneConfig()
    config.display()
    # create model in training mode
    model = modellib.MaskRCNN(
        mode="training", config=config, model_dir=MODEL_DIR)
    model = init_weights(args, model)

    ##############################################################################
    #    Training                                                                #
    ##############################################################################
    # Train in two stages:
    # 1. Only the heads. Here we're freezing all the backbone layers and training
    # only the randomly initialized layers (i.e. the ones that we didn't use
    # pre-trained weights from MS COCO). To train only the head layers, pass
    # layers='heads' to the train() function.
    #
    # 2. Fine-tune all layers. For this simple example it's not necessary, but we're
    #  including it to show the process. Simply pass layers="all to train all layers.
    train(args, model)
