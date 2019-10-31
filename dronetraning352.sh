#!/bin/bash

# 'select' set number node
# 'ncpus' set number cpu
# 'mem' set RAM value * node = total
#PBS -l select=8:ncpus=4:mem=16gb

# set run queue

# specify qgpu resource
#PBS -q common_gpuQ

# set maximum exectuion time
#PBS -l walltime=1500:00:0

# set mail
#PBS -M francesco.argentieri@studenti.unitn.it

# send mail when start and end job
#PBS -m be
# write error log
#PBS -e error_dronelanding352.log

# write out log
#PBS -o result_dronelanding352.log

# show module
module avail

# load moudle
module load python-3.5.2 cuda-9.0

# show loaded module
module list

##############################################################################
# setup environmnet
##############################################################################
source $PWD/myenv/bin/activate
echo
echo "Updating source . . ."
rm -rf $(find $PWD -name "ProjectThesis")
git clone -b develop https://github.com/frank1789/ProjectThesis.git
echo "Done"
echo
cd ProjectThesis

##############################################################################
# Prepare dataset
##############################################################################
# Download dataset
python3 downloadmanager.py

# clean from from garbage files
unzip -qq dataset.zip
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

##############################################################################
# deactivate virtual env
##############################################################################
deactivate

