# Human-Based Fall Detection

This project investigates the impact of differing clip lengths on vision model performance for fall detection, utilizing multiple datasets and advanced machine learning models. It aims to optimize real-world applications of fall detection systems, emphasizing trade-offs between accuracy, computational efficiency, and response time.

## Overview
Falls pose a significant health risk, especially for the elderly, often leading to severe injuries or fatalities. This project focuses on developing and evaluating state-of-the-art vision-based fall detection systems, analyzing how clip length impacts performance. Key datasets and machine learning models, such as MMAction2, VideoMAE V2, and Uniformer V2, are employed to study these trade-offs.

## Key Features
- **Dataset Utilization:** RFDS, HQFDS, and other datasets tailored for simulated and real-world scenarios.
- **Model Evaluation:** Detailed performance analysis of VideoMAE V2 and Uniformer V2 using metrics like accuracy, precision, and recall.
- **Resource Efficiency:** Exploration of computational trade-offs between frame lengths, model architecture, and GPU usage.
- **Real-World Relevance:** Highlights the balance between detection accuracy and real-time response requirements.

## Datasets
### Overview
- **RFDS (Real-World Fall Dataset):** Contains 120 fall sequences with diverse scenarios (e.g., indoor, outdoor, sparse, and dense).
- **HQFDS (High-Quality Fall Dataset):** Features 55 fall sequences captured from multiple camera angles.

### Preprocessing
- Clips are generated using a sliding window method with frames sampled at various lengths (15, 25, 35, 50).
- Clips are resized to 224x224 pixels and augmented using random flipping.
- Training/validation split: 80% training, 20% validation.

## Models
### VideoMAE V2
- A video representation learning model utilizing dual masking strategies for efficient pre-training.
- Pretrained on the Kinetics-400 dataset.
- Optimized using the AdamW optimizer with cosine annealing.

### Uniformer V2
- Combines convolutional and self-attention mechanisms for efficient video understanding.
- Pretrained using the CLIP-400 dataset.
- Demonstrates high performance in benchmarks, achieving a balance between accuracy and computational efficiency.

## Installation and Usage
### Generating Data
1. Download datasets and place them in the appropriate directories:
   - `Data/{DATASET}/originals`
2. Run the Jupyter notebook corresponding to the desired model pipeline.

### Running MMAction2 Models
1. Install the MMAction2 framework following the [official documentation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html).
2. Place configuration, SLURM, and data files as follows:
   - `_base_.py` → `{mmaction2_root}/configs/_base_/`
   - `UniformerV2` → `{mmaction2_root}/configs/recognition/uniformerv2/`
   - `VideoMAEV2` → `{mmaction2_root}/configs/recognition/videomaev2/`
   - SLURM files → `{mmaction2_root}/`
   - RFDS and HQFDS data → `{mmaction2_root}/data/`
3. Use SLURM scripts or MMAction2 commands to train and evaluate models.

### Running the Optical Flow Neural Network
1. Set up the Python environment using `environment.txt`.
2. Verify the `root_dir` parameter in `train.py` points to the dataset directory.
3. Run `train.py` to train the network.

## File Structure
### Data
- Each dataset folder contains:
  - `originals`: Raw data.
  - `model_Pipeline.ipynb`: Jupyter notebook for preprocessing.
  - `model`: Processed data ready for model input.
  - `labels.xxx`: Labels with fall start and end times.

### Models
- Separate subdirectories for each model containing:
  - Configuration files
  - Training scripts

## Results
- **Inference Times:** VideoMAE V2 demonstrates lower inference times across frame lengths compared to Uniformer V2, making it more suitable for real-time applications.
- **Accuracy Trends:** Uniformer V2 achieves higher accuracy, particularly with longer clip lengths (50 frames).
- **Performance Trade-offs:** The project highlights the balance between accuracy, recall, and computational overhead.

## Future Directions
- Explore advanced sampling methods to improve efficiency.
- Investigate additional preprocessing techniques like pose estimation and optical flow.
- Combine datasets for a two-step training strategy to generalize and fine-tune models for specific scenarios.

## References
A detailed report of the methodology, results, and related work can be found in the [REPORT.pdf](./REPORT.pdf) file.

---
For more information or questions, feel free to open an issue or contact the project contributors.

