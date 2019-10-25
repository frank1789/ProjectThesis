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
#PBS -e error_dronelanding.log

# write out log
#PBS -o result_dronelanding.log

# show module
module avail

# load moudle
module load python-3.5.2 cuda-9.0

# show loaded module
module list

echo
echo --- launch Python3 script --- ###
echo

# check package
echo
echo -------------------------------------
echo display avaiable package
echo
pip3 freeze list
echo
echo -------------------------------------
echo

echo "Serching for datasetfolder"
datasetpath=$(find $PWD -name "landingzone")
echo "found dataset folder at ${datasetpath}"
echo "Searching for annotations file"
datasetannotations=$(find $PWD -name "annotations.json")
echo "found dataset annotaions at ${datasetannotations}"

# update src
echo "Updating source . . ."
src=$(find $PWD -name "ProjectThesis")
rm -rf src
git clone -b develop https://github.com/frank1789/ProjectThesis.git
echo "Done"
echo
# launch script python
# environment variable
export HDF5_USE_FILE_LOCKING=FALSE
python3 /home/francesco.argentieri/ProjectThesis/landingzone.py ${datasetannotations} ${datasetpath}
