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
module load python-3.5.2 cuda-9.0 singularity

# show loaded module
module list

# update src
echo
echo "Updating source . . ."
src=$(find $PWD -name "ProjectThesis")
rm -rf src
git clone -b develop https://github.com/frank1789/ProjectThesis.git
echo "Done"
echo
# environment variable
export HDF5_USE_FILE_LOCKING=FALSE
cd ProjectThesis
singularity exec --nv docker:// sos / myimage
python3 /home/francesco.argentieri/ProjectThesis/landingzone.py ${datasetannotations} ${datasetpath}
