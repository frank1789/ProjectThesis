#!usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import signal
import sys
import warnings

import numpy as np
from PIL import Image, ImageDraw

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config
from imgaug import augmenters as iaa
from statisticanalysis import HistoryAnalysis

warnings.filterwarnings('ignore', category=FutureWarning)

# Root directory of the project
ROOT_DIR = os.getcwd()

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


##############################################################################
# Configuration
##############################################################################

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
    NUM_CLASSES = 1 + 3  # background + landing mate

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 50

    STEPS_PER_EPOCH = 1


##############################################################################
# Dataset
##############################################################################

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

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "landing zone":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


##############################################################################
# Function
##############################################################################
def train(args, model) -> None:
    """
    Train the model

    Parameters
    ----------
    :param args:
        (object) Inputs the parameters passed to the command-line script as input.
    :param model:
        (object) model of the neural network
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

    # Image augmentation
    # https://imgaug.readthedocs.io/en/latest/source/examples_basics.html
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 1.0),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's channel with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print("Training network heads")
    hist1 = model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=1,
                        layers='heads',
                        augmentation=seq)

    print("Train all layers")
    hist2 = model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 10,
                        epochs=1,
                        layers='all',
                        augmentation=seq)

    HistoryAnalysis.plot_history(hist1, "head_lz")
    HistoryAnalysis.plot_history(hist2, "all_lz")

    # # visualize
    # dataset = dataset_train
    # image_ids = np.random.choice(dataset.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #     visualize.display_top_masks(
    #         image, mask, class_ids, dataset.class_names)


def init_weights(args, model) -> object:
    """
    Initializes the neural network weights from known networks such as COCO, imagenet or from the last training.

    Parameters
    ----------
    :param args:
        (object) Inputs the parameters passed to the command-line script as input --> weights
    :param model:
        (object) model of the neural network.

    Returns
    -------
    :return model:
        (object) returns the model with the weights to be trained.
    """
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
