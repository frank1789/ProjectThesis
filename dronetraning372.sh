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
#PBS -e error_dronelanding372.log

# write out log
#PBS -o result_dronelanding372.log

# show module
module avail

# load moudle
module load python-3.7.2 cuda-9.0

# show loaded module
module list

##############################################################################
# setup environmnet
##############################################################################
echo
echo "Updating source . . ."
rm -rf $(find $PWD -name "ProjectThesis")
git clone -b develop https://github.com/frank1789/ProjectThesis.git
echo "Done"
echo
cd ProjectThesis
echo "Enter project folder: " $PWD
echo "Create folder for virtual environment"
mkdir virtualproject
cd virtualproject
echo "Enter virtual env folder: " $PWD
python3 -m venv project_venv
source project_venv/bin/activate
cd ..
echo "Enter project folder: " $PWD
echo "Install requirements . . ."
which pip
which python3
echo
echo "list pip before install:"
pip list 
echo
pip install -r requirements.txt
pip install -U requests
python -m pip install --user scipy
echo "list pip after install:"
pip list

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
echo "start training"
#python3 setup.py install
# environment variable
export HDF5_USE_FILE_LOCKING=FALSE
# launch script python
python3 landingzone.py -a ${datasetannotations} -d ${datasetpath} --weights=coco

##############################################################################
# deactivate virtual env
##############################################################################
deactivate

