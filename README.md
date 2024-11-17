
# ConvLSTM-UNet Framework for Fireline Prediction

This repository implements a framework for using **ConvLSTM-UNet** models to predict fireline behavior and dynamics, incorporating K-Fold training, post-training evaluation, and visualization.

![Wildfire Simulation](https://github.com/adibgpt/Wildfire-Propagation/blob/ba5ff2c277f8fc9ff71bfeb5a7856ad9f5846305/wildfires_firebench_Dataset.width-1250.png)
---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
   - [Initialization](#initialization)
   - [Training](#training)
   - [Post-Training Prediction](#post-training-prediction)
   - [Visualization](#visualization)
4. [Usage](#usage)

---

## Overview

The framework combines **ConvLSTM** (for temporal features) and **UNet** (for spatial resolution enhancement) to process multi-channel datasets. It performs:

- **K-Fold Cross-Validation**: To assess model robustness.
- **Post-Training Prediction**: For long-term fireline dynamics.
- **Visualization**: Includes animations of fireline prediction and difference plots.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/adibgpt/Wildfire-Propagation.git
   cd Wildfire-Propagation
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Components

### Initialization

The code initializes key hyperparameters and prepares the dataset for use:

- Variables like `sequence_length`, `n_channels`, and `nx`, `ny` for spatial resolution are set.
- GPU acceleration is used with CUDA where available.

### Training

The `k_fold_training.py` script implements the training loop:

- **K-Fold Cross-Validation** ensures robust training by splitting the data into `n_splits` folds.
- Mixed-precision training is enabled using `torch.cuda.amp` for faster computations.
- Metrics like MSE, SSIM, and PSNR are tracked per epoch and fold.

![MSE](https://github.com/adibgpt/Wildfire-Propagation/blob/062ebcc029740a0ac27b56b073d99cf54014dc21/Images/MSE.png)
![PSNR](https://github.com/adibgpt/Wildfire-Propagation/blob/8c0f3b5fee274a66386083e095a0e2204a759421/Images/PSNR.png)
![SSIM](https://github.com/adibgpt/Wildfire-Propagation/blob/c78b779c123776808b50d09ab0c2d5abd755061f/Images/SSIM.png)

### Post-Training Prediction

`post_training_predictions.py` handles predictions and includes:

- **Custom Loss Functions**:
  - `STE_Loss`: Computes the sum difference error.
  - `JaccardSimilarity`: Measures IoU for fireline data.
- Long-term predictions based on input sequences.
- Inverse normalization for real-world interpretations.

### Visualization

The `result_visualization.py` script generates:

- **Metric Plots**:
  - MSE, SSIM, and PSNR trends.
- **Difference Plots**:
  - Visualize deviations between predictions and ground truth.
- **Fireline Animation**:
  - Animated fireline progression over time.

![Fireline](https://github.com/adibgpt/Wildfire-Propagation/blob/a38ec4e5f142265229e99a0cdd5086727b3a1201/Visualization/u8_ramp7.5_filt8_fireline_prediction.gif)
![Fireline](https://github.com/adibgpt/Wildfire-Propagation/blob/de467e656c0d898856fc98424a0d2782fbe13889/Visualization/u8_ramp15_filt8_fireline_prediction.gif)
![Fireline](https://github.com/adibgpt/Wildfire-Propagation/blob/d17cc25174bffd74f039142dae67c2626768e34e/Visualization/u18_ramp2.5_filt8_fireline_prediction.gif)

---

## Usage

### Training the Model

Run the training script:

```bash
python k_fold_training.py
```

This performs K-Fold cross-validation and saves the trained models and evaluation metrics.

### Post-Training Predictions

Make predictions and save results:

```bash
python post_training_predictions.py
```

### Visualizing Results

Generate metric plots and animations:

```bash
python result_visualization.py
```

### Example Fireline Prediction

Example animations and plots are saved as `.gif` and `.png` files in the project directory.

---

## File Structure

```plaintext
Wildfire-Propagation
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   ├── clear_cuda.py
│   ├── conv_lstm_unet.py
│   ├── data_preprocessing.py
│   ├── init_variables.py
│   ├── k_fold_training.py
│   ├── post_training_evaluation.py
│   ├── post_training_fireline_evaluation.py
│   ├── post_training_predictions.py
│   └── result_visualization.py
```

---

## Example Results

- **Fireline Animation**: Shows predicted and ground truth firelines over time.
- **MSE and SSIM Plots**: Evaluate model performance across epochs.

---

## License

This repository is open-source and available under the MIT license.
