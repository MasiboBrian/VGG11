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

export dest_path_model=${SCRATCH_HOME}/vgg11_bn_trad_aug

export dest_path_data=${SCRATCH_HOME}/vgg11_bn_trad_aug/data

mkdir -p ${dest_path_data}

#####################################################################################################################################
echo "Moving all model files"

rsync --archive --update --compress /home/${STUDENT_ID}/Code/Cluster/vgg11_bn_trad_aug/*.py ${dest_path_model}/

echo "done"

echo "moving data"

rsync --archive --update --compress /home/${STUDENT_ID}/Code/Cluster/data/cifar-10-batches-py ${dest_path_data}/
#rsync --archive --update --compress /home/${STUDENT_ID}/Code/Cluster/data/Sw@ ${dest_path_data}/

echo "done. verifying."

ls ${dest_path_data}/

echo "done."

#######################################################################################################################################
# Activate the relevant virtual environment:
echo "Activating conda environment"
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ${SCRATCH_HOME}

########################################################################################################################################

cd ${dest_path_model}

echo "running model"
python train_network.py --num_epochs 100 --learning_rate 0.001 --weight_decay 0.0001

echo "Done."

#########################################################################################################################################
#Move data back to DFS

echo "Moving results back to DFS"

mkdir -p /home/${STUDENT_ID}/Code/Cluster/vgg11_bn_trad_aug/Results/

rsync --archive --update --compress /disk/scratch/s2089883/vgg11_bn_trad_aug/exp_1/result_outputs /home/${STUDENT_ID}/Code/Cluster/vgg11_bn_trad_aug/Results/
rsync --archive --update --compress /disk/scratch/s2089883/vgg11_bn_trad_aug/exp_1/saved_models/train_model_latest /home/${STUDENT_ID}/Code/Cluster/vgg11_bn_trad_aug/Results/

echo "Done."

echo "deleting generated files"

rm -r /disk/scratch/s2089883/vgg11_bn_trad_aug

echo "Done."
#########################################################################################################################################
