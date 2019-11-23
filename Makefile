


DIR_DATASET := $(find ${PROJECT_DIR} -name "landingzone")








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

.PHONY: prepare
prepare:
	@echo
	@echo "==> Dowload dataset"
	@echo
	# Download dataset
	coverage run downloadmanager.py
	@echo
	@echo "==> unizp dataset zip file"
	@echo
    # clean from from garbage files
	unzip -qq dataset.zip
	find . -type f -name '._*' -delete
	delfolder=$(find . -type d -name '__MACOSX')
	rm -rf ${delfolder}
	rm -rf dataset.zip
	datasetannotations=$(find $PWD -name "annotations.json")
	@echo "found dataset annotaions at ${datasetannotations}"


.PHONY: test
test: prepare
	# export csv
	coverage run -a xml_to_csv.py --annotations ${DIR_DATASET}/train -output_csv train_labels.csv
	coverage run -a xml_to_csv.py --annotations ${DIR_DATASET}/validate -output_csv validate_labels.csv
	@echo
	coverage run -a ${landing_script} train --annotations ${datasetannotations} --dataset ${DIR_DATASET} --weights=coco
  	coverage run -a ${landing_script} train --annotations ${datasetannotations} --dataset ${DIR_DATASET} --weights=imagenet

help:
	${CC} --help
