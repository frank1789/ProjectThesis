| Branch | Status |
| :----  | :----: |
| Develop build | [![Build Status](https://img.shields.io/travis/frank1789/ProjectThesis/develop)](https://travis-ci.org/frank1789/ProjectThesis)|
| Develop coverage | [![Coverage Status](https://coveralls.io/repos/github/frank1789/ProjectThesis/badge.svg?branch=develop)](https://coveralls.io/github/frank1789/ProjectThesis?branch=feature/test)|
| 3D-environment build | [![Build Status](https://img.shields.io/travis/frank1789/ProjectThesis/3D-environment)](https://travis-ci.org/frank1789/ProjectThesis)|
| 3D-environment coverage | [![Coverage Status](https://coveralls.io/repos/github/frank1789/ProjectThesis/badge.svg?branch=3D-environment)](https://coveralls.io/github/frank1789/ProjectThesis?branch=3D-environment)|

# Project
This project is based on the [Mask_RCNN](https://github.com/matterport/Mask_RCNN) code.

## Intent
This project concerns my experimental thesis for the master's degree in mechatronic engineering.
The intention is to identify the drone landing pads from different heights and positions.
These were created using 3D modeling software such as Blender. In addition, the carpets were placed in a virtual context to recreate some typical amateur flight situations.
The camera settings to generate the images are those of the Raspberry Pi Camera V2.
The result is visible in the figure.

### Enviroment test
In the [3D-environment](https://github.com/frank1789/ProjectThesis/tree/3D-environment) branch I present the python script and the models made in blender to generate some simple environments. These environments are representations of possible environments that can be fired with aerial shots from a drone that mounts the camera in a position perpendicular to the ground.
With the help of the script we tried to replicate the positions taken by a flying drone that approaches landing.
To improve the meshing of the net it was decided to use replicas using a simple day / night cycle in Blender modifying the color and intensity parameters of the light source.
The camera reproduces the features of the Raspberry Pi Camera V2 as mentioned before.

Render samples:
![sample1](https://github.com/frank1789/ProjectThesis/sample/sample1.png)
![sample2](https://github.com/frank1789/ProjectThesis/sample/sample2.png)
![sample3](https://github.com/frank1789/ProjectThesis/sample/sample3.png)


# Requirements
Python 3.5, TensorFlow 1.3, Keras 2.2.5 and other common packages listed in requirements.txt.

Install dependencies

```
pip3 install -r requirements.txt
```

Run setup from the repository root directory
```
python3 setup.py install
```
## Training
You can use those weights as a starting point to train your own variation on the network. You can import this module in Jupyter notebook (see the provided notebooks for examples) or you can run it directly from the command line as such:

```
python3 landinzone.py train --annotations=path/to/annontations.json --dataset=path/to/dataset --weights=coco
python3 landinzone.py train --annotations=path/to/annontations.json --dataset=path/to/dataset --weights=imagenet
```

### Conversion and export entire model
To export the entire model in Keras (.h5) or tenosrflow (.pb) format, it is sufficient to execute the script once the training is completed:
```
python3 convert_keras_tf.py
```
