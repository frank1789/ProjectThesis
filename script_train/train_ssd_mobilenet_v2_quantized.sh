#!usr/bin/sh

##############################################################################
# Download pre-trained model
##############################################################################
echo $PWD
remote_link_model=http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
curl -L $remote_link_model -o ssd_mobilenet_v2_quantized.tar.gz
tar -xvf ssd_mobilenet_v2_quantized.tar.gz
rm ssd_mobilenet_v2_quantized.tar.gz
cd ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
remote_config=https://raw.githubusercontent.com/frank1789/ProjectThesis/develop/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config
curl -OL $remote_config -o out_dir
export MODEL_DIR=$PROJECT_DIR/models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
cd ..
cd ..
echo $PWD

##############################################################################
# Running the Training Job
##############################################################################

export PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
export TF_RESEARCH_MODEL_DIR=$PROJECT_DIR/tf-models/research

export NUM_TRAIN_STEPS=50000
export SAMPLE_1_OF_N_EVAL_EXAMPLES=1

# From the project/tf-models/research/ directory
# Make sure you've updated PYTHONPATH
python3 ${TF_API_DIR}/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
