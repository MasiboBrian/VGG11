#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}
export SCRATCH_DISK=/disk/scratch
export SCRATCH_HOME=${SCRATCH_DISK}/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

export DATASET=CIFAR10
export MODEL=style_monet_pretrained

export dest_path_model=${SCRATCH_HOME}/CycleGAN_ref/pytorch-CycleGAN-and-pix2pix/scripts/checkpoints/${MODEL}

#####################################################################################################################################
echo "Moving all model files"

rsync --archive --update --compress /home/${STUDENT_ID}/CycleGAN_ref ${SCRATCH_HOME}/

echo "done"

#######################################################################################################################################
# Activate the relevant virtual environment:
echo "Activating conda environment"
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ${SCRATCH_HOME}

######################################################################################################################################
export BATCH=Batch1

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - CHANGE FOR EVERY RUN
export src_path_data=/home/${STUDENT_ID}/Code/${DATASET}_Data/Train/${BATCH}

# input data directory path on the scratch disk of the node - CHANGE FOR EVERY RUN
export dest_path_data=${SCRATCH_HOME}/data/${BATCH}

mkdir -p ${dest_path_data}  # make it if required

rsync --archive --update --compress ${src_path_data}/ ${dest_path_data}

echo "Done."

echo "generating fake images"
python ${SCRATCH_HOME}/CycleGAN_ref/pytorch-CycleGAN-and-pix2pix/test.py --dataroot ${dest_path_data}/ --name ${dest_path_model} --model test --no_dropout --num_test 10000
echo "Done."

echo "deleting real images"
rm /disk/scratch/s2089883/CycleGAN_ref/pytorch-CycleGAN-and-pix2pix/scripts/checkpoints/style_monet_pretrained/test_latest/images/*real.png

echo "moving fake images to ${SCRATCH_HOME}/Monet/${BATCH}/"
mkdir -p ${SCRATCH_HOME}/Monet/${BATCH}
rsync -ua --progress /disk/scratch/s2089883/CycleGAN_ref/pytorch-CycleGAN-and-pix2pix/scripts/checkpoints/style_monet_pretrained/test_latest/images/*fake.png ${SCRATCH_HOME}/Monet/${BATCH}/
rsync -ua --progress /home/${STUDENT_ID}/Code/Cluster/CycleGAN_Monet/PickleAugmented1.py ${SCRATCH_HOME}/Monet/

ls ${SCRATCH_HOME}/Monet/

echo "Pickling fake images"
cd ${SCRATCH_HOME}/Monet/
python PickleAugmented1.py

#Move data back to DFS
echo "Moving results back to DFS"

mkdir -p /home/${STUDENT_ID}/Code/Cluster/data/Raw/Monet/${BATCH}
mkdir -p /home/${STUDENT_ID}/Code/Cluster/data/Monet

rsync --archive --update --compress /disk/scratch/s2089883/CycleGAN_ref/pytorch-CycleGAN-and-pix2pix/scripts/checkpoints/style_monet_pretrained/test_latest/images/*fake.png /home/${STUDENT_ID}/Code/Cluster/data/Raw/Monet/${BATCH}
rsync --archive --update --compress /${SCRATCH_HOME}/Monet/*.pickle /home/${STUDENT_ID}/Code/Cluster/data/Monet/

echo "Done."

echo "deleting files"
rm -r ${dest_path_data}
rm -r ${SCRATCH_HOME}/Monet
rm -r ${SCRATCH_HOME}/CycleGAN_ref
echo "Done."
#########################################################################################################################################
