#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-95 -p alvis
#SBATCH --gpus-per-node=A40:1

#SBATCH --job-name=fall_detection
#SBATCH --time=100:00:00                
#SBATCH --output=UNI_25_HQFDS-_%j.log           

source /mimer/NOBACKUP/groups/ulio_inverse/human-fall-detection/myenv/bin/activate

python tools/train.py configs/recognition/uniformerv2/UNI_25_HQFDS.py
                                           