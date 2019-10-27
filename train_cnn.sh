#!usr/bin/env sh
# -*- coding: utf-8 -*-

##############################################################################
# Prepare dataset
##############################################################################
# Download dataset
python3 downloadmanager.py

# clean from from garbage files
unzip dataset.zip
find . -type f -name '._*' -delete
delfolder=$(find . -type d -name '__MACOSX')
rm -rf ${delfolder}

# search folder and files
echo "Serching for datasetfolder"
datasetpath=$(find $PWD -name "landingzone")
echo "found dataset folder at ${datasetpath}"
echo "Searching for annotations file"
datasetannotations=$(find $PWD -name "annotations.json")
echo "found dataset annotaions at ${datasetannotations}"
echo

##############################################################################
# Training
##############################################################################
python3 setup.py install
# environment variable
export HDF5_USE_FILE_LOCKING=FALSE
# launch script python
python3 landingzone.py -a ${datasetannotations} -d ${datasetpath} --weights=coco
