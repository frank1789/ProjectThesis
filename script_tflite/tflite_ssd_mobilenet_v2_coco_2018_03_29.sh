#!usr/bin/env sh

##############################################################################
# Setup enviroment
##############################################################################

export TF_CPP_MIN_LOG_LEVEL=3
export PROJECT_DIR=$PWD
export TF_API_DIR=$PROJECT_DIR/tf-models/research
export PYTHONPATH=$PYTHONPATH:$TF_API_DIR:$TF_API_DIR/slim
export MODEL_DIR=$PROJECT_DIR/models/ssd_mobilenet_v2_coco_2018_03_29
export PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config
export CHECKPOINT_PATH=$MODEL_DIR/model.ckpt-96
export OUTPUT_DIR=$MODEL_DIR/tmp/tflite

##############################################################################
# Convert to TensorFlow Lite
##############################################################################
# Run the export_tflite_ssd_graph.py script to get the frozen graph
python3 ${TF_API_DIR}/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${CHECKPOINT_PATH} \
    --output_directory=${OUTPUT_DIR} \
    --add_postprocessing_op=true \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops

cd tf-models/tensorflow
bazel run -c opt tensorflow/lite/toco:toco -- \
    --input_file=${OUTPUT_DIR}/tflite_graph.pb \
    --output_file=${OUTPUT_DIR}/detect.tflite \
    --input_shapes=1,512,512,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops \
    --default_ranges_min=0 --default_ranges_max=6
