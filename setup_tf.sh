#!usr/bin/env sh

##############################################################################
# Setup enviroment
##############################################################################
export TF_CPP_MIN_LOG_LEVEL=3
export PROJECT_DIR=$PWD
export TF_API_DIR=$PROJECT_DIR/tf-models/research
export PYTHONPATH=$PYTHONPATH:$TF_API_DIR:$TF_API_DIR/slim

echo
echo "Project directory: ${PROJECT_DIR}"
echo "API directory: ${TF_API_DIR}"
echo

# execute test
python3 $TF_API_DIR/object_detection/builders/model_builder_test.py

##############################################################################
# Prepare dataset
##############################################################################
DIR_DATASET=$(find ${PROJECT_DIR} -name "landingzone")
echo
if [ ! -d "$DIR_DATASET" ]; then
    echo
    echo "==> Dowload dataset"
    echo
    # Download dataset
    python3 downloadmanager.py
    #
    echo
    echo "==> unizp dataset zip file"
    echo
    # clean from from garbage files
    unzip -qq dataset.zip
    find . -type f -name '._*' -delete
    delfolder=$(find . -type d -name '__MACOSX')
    #
    echo "delete files unnecessary"
    echo
    rm -rf ${delfolder}
    rm -rf dataset.zip
    # make dataset
    DIR_DATASET=$(find ${PROJECT_DIR} -name "landingzone")
    python3 preparedata.py -d $DIR_DATASET -s 30
    # export csv
    python3 xml_to_csv.py -a $DIR_DATASET/train -o train_labels.csv
    python3 xml_to_csv.py -a $DIR_DATASET/validate -o validate_labels.csv
    echo
    # create records
    CSV_TRAIN=$(find . -name 'train_labels.csv')
    CSV_VALIDATE=$(find . -name 'validate_labels.csv')
    DIR_TRAIN=$(find $DIR_DATASET -name "train")
    DIR_VALIDATE=$(find $DIR_DATASET -name "validate")
    python3 generate_tfrecord.py \
        --csv_input=${CSV_TRAIN} \
        --output_path=data/train.record \
        --img_path=${DIR_TRAIN}
    python3 generate_tfrecord.py \
        --csv_input=${CSV_VALIDATE} \
        --output_path=data/validate.record \
        --img_path=${DIR_VALIDATE}
    echo
fi

##############################################################################
# Download pre-trained model
##############################################################################

mkdir models
cd $PROJECT_DIR/models
if [ "$1" = coco_quantized ]; then
    sh $PROJECT_DIR/script_train/train_ssd_mobilenet_v2_quantized.sh

elif [ "$1" = mask_inception ]; then
    remote_link_model=http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
    curl -L $remote_link_model -o mask_rcnn_inception_v2.tar.gz
    tar -xvf mask_rcnn_inception_v2.tar.gz
    rm mask_rcnn_inception_v2.tar.gz
    cd mask_rcnn_inception_v2_coco_2018_01_28
    echo $PWD
    remote_config=https://raw.githubusercontent.com/frank1789/ProjectThesis/develop/models/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config
    curl -OL $remote_config -o out_dir
    rm checkpoint
    export MODEL_DIR=$PROJECT_DIR/models/mask_rcnn_inception_v2_coco_2018_01_28

elif [ "$1" = ssd_coco ]; then
    sh $PROJECT_DIR/script_train/train_ssd_mobilenet_v2_coco_2018_03_29.sh

elif [ "$1" = keras_mask_rcnn ]; then
    cd ..
    sh $PROJECT_DIR/script_train/train_keras_mask_rcnn.sh

else
    echo "Wrong or missing argument pass to script."
    exit 0
fi

exit 0
