#!usr/bin/env sh

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
# launch script python
export HDF5_USE_FILE_LOCKING=FALSE
python3 landingzone.py train -a ${datasetannotations} -d ${datasetpath} --weights=coco
