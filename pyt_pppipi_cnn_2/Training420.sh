#! /bin/bash

######## Part 1 #########
# Script parameters     #
#########################

# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu

# Specify the QOS, mandatory option
#SBATCH --qos=normal

# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=mlgpu

# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=420pyt_pppipi_cnn_1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_2/%j.Training.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=50000

# Specify how many GPU cards to use
#SBATCH --gres=gpu:1

######## Part 2 ######
# Script workload    #
######################
# list the allocated hosts
srun -l hostname

# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

# Cuda info
time(python ./TestCuda.py)

# Graph Formation
#time(python ./GraphDef.py)

# Loading data
#time(python ./DataLoad.py)

# Defining the model
#time(python ./Model.py)

# Training 
time(python ./Training420.py)
