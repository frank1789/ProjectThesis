#!usr/bin/env sh

##############################################################################
# Setup enviroment
##############################################################################

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

    echo
    echo "==> unizp dataset zip file"
    echo
    # clean from from garbage files
    unzip -qq dataset.zip
    find . -type f -name '._*' -delete
    delfolder=$(find . -type d -name '__MACOSX')

    echo "delete files unnecessary"
    echo
    rm -rf ${delfolder}
    rm -rf dataset.zip
fi

# export csv
python3 xml_to_csv.py -a $DIR_DATASET/train -o train_labels.csv
python3 xml_to_csv.py -a $DIR_DATASET/validate -o validate_labels.csv
echo

# create records
csv_train=$(find . -name 'train_labels.csv')
csv_val=$(find . -name 'validate_labels.csv')
DIR_TRAIN=$(find $DIR_DATASET -name "train")
DIR_VALIDATE=$(find $DIR_DATASET -name "validate")
sh tf.sh generate_tfrecord.py --csv_input=${csv_train} \
    --output_path=data/train.record \
    --img_path=${DIR_TRAIN}
sh tf.sh generate_tfrecord.py --csv_input=${csv_val} \
    --output_path=data/validate.record \
    --img_path=${DIR_VALIDATE}
echo

##############################################################################
# Download pre-trained model
##############################################################################

cd $PROJECT_DIR/models
curl -L http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz -o ssd_mobilenet_v2_quantized.tar.gz
tar -xvf --exclude='pipeline.config' ssd_mobilenet_v2_quantized.tar.gz
rm ssd_mobilenet_v2_quantized.tar.gz
cd ..

##############################################################################
# Running the Training Job
##############################################################################

export MODEL_DIR=$PROJECT_DIR/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
export PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
export TF_RESEARCH_MODEL_DIR=$PROJECT_DIR/tf-models/research

export NUM_TRAIN_STEPS=50000
export SAMPLE_1_OF_N_EVAL_EXAMPLES=1

# From the project/tf-models/research/ directory
# Make sure you've updated PYTHONPATH
sh tf.sh ${TF_API_DIR}/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
