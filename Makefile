
# Directives of the interpreter, path to the python file
# command to carry out the conversion
DEVPATH := /opt/intel/openvino_2019.3.334/deployment_tools/model_optimizer
INTERPRETER := python3
CONVERT := mo_tf.py
SUMMARIZE := mo/utils/summarize_graph.py
CC := ${INTERPRETER} ${DEVPATH}/${COMMAND}

# Look for the neural network model file in * .pb format
MODEL_FREEZED := $(wildcard **/*.pb)
# Look for the neural network model log
LOGBOARD := logs/landing20191105T1206/events.out.tfevents.1572952039.hpc-g01-node01.unitn.it
INPUT_MODEL := --input_model ${MODEL_FREEZED}

# Optional topics to pass to the conversion script
FLAG_INPUT_LAYER := --input input_image
FLAG_OUTPUT_LAYER := --output output_mrcnn_mask
TENSOR_SHAPE := "(1,512,512,3)"
FLAG_SHAPE := --input_shape ${TENSOR_SHAPE}
FLAG_DATA_TYPE := --data_type FP16

# TensorBoard Flags
FLAG_TENSORBOARD := --tensorboard_logdir ${LOGBOARD}
FLAGS := ${FLAG_SHAPE} ${FLAG_INPUT_LAYER} ${FLAG_DATA_TYPE} ${FLAG_TENSORBOARD}

.PHONY: all

all: export summarize
	${CC}${CONVERT} ${INPUT_MODEL} ${FLAGS}

summarize:
	${CC}${SUMMARIZE} ${INPUT_MODEL}

export:
	${INTERPRETER} convert_keras_tf.py

help:
	${CC} --help
