# Initialization and variable setup script for the ConvLSTM-UNet framework

import numpy as np
import random
import torch
from sklearn.model_selection import KFold

# Set deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Control variables
use_all_cpus = False

# Paths and dataset configurations
folder = "/kaggle/input/flame-ai-8-m-filter-agl-1-5"
u_trainval_ = [6, 10, 14]
ramp_trainval_ = [0.0, 5, 10.0]

u_test_ = [8, 12, 18]
ramp_test_ = [2.5, 7.5, 15]

num_epochs = 200
x_extract_every_array = 8
y_extract_every_array = 8
zag_ind = 4  # Index of zag to use as inputs

# Data sampling configurations
epoch_train_percentage = 1.0  # Randomly sample the data for training and change the samples every epoch
remove_data_percentage = 0  # Randomly remove data to fit in Kaggle

# Cross-validation configurations
n_splits = 7  # Number of K-Fold splits

# Stability prediction settings
randomize_stability_predictions = True
at_least_n_predictions = 10

# Model configurations
model_defined = False
power_of_two = False

# Select velocity type
ustar_or_u10 = 'ustar'
if ustar_or_u10 == 'u10':
    raise NotImplementedError("u10 is not yet supported")

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Generate train-validation split
u_trainval = np.repeat(u_trainval_, len(ramp_trainval_)).tolist()
ramp_trainval = np.tile(ramp_trainval_, len(u_trainval_)).tolist()
combined = list(zip(u_trainval, ramp_trainval))
random.shuffle(combined)

# Set up K-Fold Cross-Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# Print splits for verification
for fold, (train_index, test_index) in enumerate(kf.split(combined)):
    train_data = [combined[i] for i in train_index]
    test_data = [combined[i] for i in test_index]
    u_train, ramp_train = zip(*train_data)
    u_val, ramp_val = zip(*test_data)
    
    print(f"Fold {fold + 1}")
    print(f"Training u: {list(u_train)}")
    print(f"Training ramp: {list(ramp_train)}")
    print(f"Validation u: {list(u_val)}")
    print(f"Validation ramp: {list(ramp_val)}")

# Calculate the test set
u_test = np.repeat(u_test_, len(ramp_test_)).tolist()
ramp_test = np.tile(ramp_test_, len(u_test_)).tolist()

print(f"Test u: {u_test}")
print(f"Test ramp: {ramp_test}")
