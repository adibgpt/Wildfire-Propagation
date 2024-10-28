
# Fire Prediction Model using ConvLSTM-UNet

This repository contains code for a fire line prediction project that utilizes a ConvLSTM-UNet model to predict and visualize fire progression over time. The project includes code for training with K-fold cross-validation, model evaluation, and visualization of predicted and ground truth fire lines.

![Flow Field Prediction](https://github.com/adibgpt/Super-resolution-Turbulence/blob/48f9ead8d55294f7d7d36cdf2dcff28fc7de72dc/2D%20Snapshot.png)


## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [License](#license)

---

### Project Structure

The project is divided into modular scripts for better organization and usability:

- **`data_processing.py`**: Contains functions for loading and preprocessing dataset files.
- **`model.py`**: Defines the ConvLSTM-UNet model architecture.
- **`training.py`**: Handles training using K-fold cross-validation and logs metrics.
- **`prediction.py`**: For model predictions and metrics calculation.
- **`visualization.py`**: Generates plots and animations, including the prediction vs. ground truth fire line animations.
- **`config.yaml`** (optional): Stores configuration parameters for easy adjustments.

Additional directories:
- **`data/`**: Place your dataset files here.
- **`results/plots/`**: Contains metric plots generated during training and testing.
- **`results/animations/`**: Stores generated animations of fire line predictions.

---

### Installation

To run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fire-prediction-model.git
   cd fire-prediction-model
   ```

2. **Install dependencies**:
   Install the required Python packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

The project is designed to be run in stages. Each script handles a specific part of the pipeline. Below is the typical order of execution:

1. **Data Processing**:
   Run `data_processing.py` to preprocess the data. Ensure that your dataset is placed in the `data/` directory.

   ```bash
   python data_processing.py
   ```

2. **Model Training**:
   Use `training.py` to train the model with K-fold cross-validation. Metrics are saved in the `results/plots/` directory.

   ```bash
   python training.py
   ```

3. **Prediction**:
   Run `prediction.py` to make predictions on the test set and generate evaluation metrics.

   ```bash
   python prediction.py
   ```

4. **Visualization**:
   Use `visualization.py` to create visualizations of the predicted fire lines and other metrics.

   ```bash
   python visualization.py
   ```

---

### Configuration

To customize parameters like sequence length, number of epochs, batch size, and data paths, modify the **`config.yaml`** file (if included) or edit these variables directly in the scripts.

---

### Output

- **Plots**: Saved in `results/plots/` and include metrics like MSE, SSIM, and PSNR.
- **Animations**: Stored in `results/animations/` and display predicted vs. ground truth fire lines and other visualizations.

---

### License

This project is licensed under the MIT License. See `LICENSE` for more information.
