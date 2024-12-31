# Human Based Fall Detection
Analysing differing clip lengths impact on model performance utilising multiple datasets and multiple models

## Models & Datasets
### Generating Data
- Download data from location file
- Place in Data/{DATASET}/originals directory
- Run jupyter notebook cells for corresponding model pipeline adjusting parameters as necessary

## Running Models
### MMAction2 Models
- Install MMACTION2 framework and environment following https://mmaction2.readthedocs.io/en/latest/get_started/installation.html
- Place config,slurm and data files in following directories
    - _base_.py -> {mmaction2_root}/configs/_base_/
    - UniformerV2 -> {mmaction2_root}/configs/recognition/uniformerv2/
    - VideoMAEV2 -> {mmaction2_root}/configs/recogntion/videomaev2/
    - Slurm files -> {mmaction2_root}/
    - RFDS data -> {mmaction2_root}/data/rfds/
    - HQFDS data -> {mmaction2_root}/data/hqfds/
- Use slurm files/framework to run experiments

### Optical Flow Neural Network
- Set up python environment using environment.txt
- Ensure dataset is correctly pointed to with root_dir parameter in train.py
- Run train.py

## File Structure
### Data
- Within the Data directory there is a subdirectory which corresponds to each data set
- Within each subdirectory is:
    - originals: Directory containing original data
    - model_Pipeline.ipynb: Notebook to generate data in required for for model
    - model: Directory containing data in the form required for the model
    - labels.xxx: Labels containing falls start and end times

### Models
- Within the Models directory there is a subdirectory which corresponds to each model
