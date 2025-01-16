#!/bin/bash
#SBATCH -J Tarsier7B_Inference # job name
#SBATCH -p gpu # partition name
#SBATCH --gpus 1 # gpu count
#SBATCH --mem-per-gpu 20G # memory per gpu

module load cuda
python inference_tarsier7b.py # script para hacer la inferencia con Tarsier-7b
module unload cuda
