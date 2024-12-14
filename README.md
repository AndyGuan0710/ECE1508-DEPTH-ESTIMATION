# Depth Estimation with U-Net: Static and Video Depth Prediction

This project implements a U-Net-based model for single-image and video depth estimation using the NYU Depth V2 dataset. The model predicts depth maps for static RGB images and ensures temporal consistency across video frames through novel loss functions. The project is divided into two main parts: **Part 1** (static depth estimation) and **Part 2** (video depth estimation with temporal constraints).

---

## Features
- **Static Depth Estimation**: Predict depth maps for individual RGB images.
- **Video Depth Estimation**: Ensure smooth and consistent depth predictions across video frames by incorporating temporal consistency loss.
- **Edge-Aware Smoothness**: Enhance depth predictions by preserving edge details while maintaining smooth transitions in uniform regions.
- **Metrics and Visualization**: Generate loss curves and evaluate temporal consistency for video predictions.

---

## Dataset

The project uses the **NYU Depth V2** dataset, which can be downloaded [here](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).  
Download and place the `nyu_depth_v2_labeled.mat` file in the root directory before running the code.

---

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed along with the following dependencies:
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- tqdm
- Jupyter Notebook (optional, for running `.ipynb` files)

Install dependencies using:
```bash
pip install -r requirements.txt
```
---

## Training

### Part 1: Static Depth Estimation

To train the U-Net model for single-image depth estimation:

1. Open the `part1.ipynb` notebook in Google Colab.
2. Ensure the runtime is set to use a **T4 GPU**:
   - Go to `Runtime > Change runtime type`.
   - Set `Hardware accelerator` to `GPU`.
3. Execute the cells to:
   - Load the NYU Depth V2 dataset.
   - Train the model for 30 epochs.
   - Save intermediate model weights (`model_epoch_#.pth`) to the working directory.

---

### Part 2: Video Depth Estimation

To train the U-Net model with temporal loss for video depth prediction:

1. Open the `part2.ipynb` notebook in Google Colab.
2. Ensure the runtime is set to use a **T4 GPU**:
   - Go to `Runtime > Change runtime type`.
   - Set `Hardware accelerator` to `GPU`.
3. Execute the cells to:
   - Load paired consecutive frames created via augmentation.
   - Train the model for 30 epochs with temporal consistency and edge-aware smoothness losses.
   - Save intermediate model weights to the working directory.

---

## Evaluation and Analysis

To evaluate and analyze the performance of the trained models:

1. Open the `Metrics&predict-video.ipynb` notebook in Google Colab.
2. Ensure the runtime is set to use a **L4 GPU**:
   - Go to `Runtime > Change runtime type`.
   - Set `Hardware accelerator` to `GPU`.
3. Execute the cells to:
   - Evaluate the trained models using metrics like MAE, RMSE, and SSIM.
   - Visualize depth predictions for static RGB images and video frames.
   - Plot training and validation loss curves.
   - Compare temporal loss across consecutive video frames.
