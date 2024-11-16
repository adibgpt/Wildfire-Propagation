# K-Fold training and evaluation script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np
import random

# Weighted MSE Loss function
def weighted_mse_loss(output, target):
    mse_loss = (output - target) ** 2
    weights = 100 * (1 - torch.tanh(3 * target))
    return torch.mean(weights * mse_loss)

# Gradient Loss function
def gradient_loss(output, target, end_channel_index):
    grad_x_output, grad_y_output = torch.gradient(output[:, :end_channel_index, :, :], dim=[2, 3])
    grad_x_target, grad_y_target = torch.gradient(target[:, :end_channel_index, :, :], dim=[2, 3])
    return F.mse_loss(grad_x_output, grad_x_target) + F.mse_loss(grad_y_output, grad_y_target)

# Initialize metrics
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()

# K-Fold Cross-Validation
kf = KFold(n_splits=7, shuffle=True, random_state=0)

# Training loop
def train_model(model, dataloader, optimizer, scaler, num_epochs, device, n_gpu):
    all_loss_epoch = []
    all_ssim_epoch = []
    all_psnr_epoch = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ssim = 0.0
        epoch_psnr = 0.0
        num_batches = len(dataloader)

        for data_seq_batch, data_targets_batch, _ in dataloader:
            optimizer.zero_grad()
            
            with autocast():  # Mixed precision training
                hidden = model.module.init_hidden(data_seq_batch.size(0)) if n_gpu > 1 else model.init_hidden(data_seq_batch.size(0))
                output = model(data_seq_batch.to(device), hidden)
                output = output.view_as(data_targets_batch.to(device))
                loss = weighted_mse_loss(output, data_targets_batch.to(device))
                ssim = ssim_metric(output[:, :3, :, :], data_targets_batch[:, :3, :, :].to(device))
                psnr = psnr_metric(output[:, :3, :, :], data_targets_batch[:, :3, :, :].to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_ssim += ssim.item()
            epoch_psnr += psnr.item()

        avg_loss = epoch_loss / num_batches
        avg_ssim = epoch_ssim / num_batches
        avg_psnr = epoch_psnr / num_batches

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")
        all_loss_epoch.append(avg_loss)
        all_ssim_epoch.append(avg_ssim)
        all_psnr_epoch.append(avg_psnr)

    return all_loss_epoch, all_ssim_epoch, all_psnr_epoch

# Main training script
if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 200
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    for fold, (train_index, test_index) in enumerate(kf.split(combined)):
        print(f"Starting fold {fold + 1}/7")
        
        # Model initialization
        model = ConvLSTM_UNet(
            input_size=(nx, ny),
            input_channels=n_channels,
            output_channels=n_channels,
            hidden_dim=64,
            kernel_size=3,
            num_layers=2,
            device=device
        ).to(device)
        if n_gpu > 1:
            model = nn.DataParallel(model)
        
        # Optimizer and scaler
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()

        # Prepare training data
        data_sequences = torch.stack([...])  # Replace with your sequence preparation
        data_targets = torch.stack([...])   # Replace with your target preparation

        dataset = TensorDataset(data_sequences, data_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Train model
        loss_epoch, ssim_epoch, psnr_epoch = train_model(
            model, dataloader, optimizer, scaler, num_epochs, device, n_gpu
        )

        # Save fold results
        torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")
        np.save(f"loss_epoch_fold_{fold + 1}.npy", loss_epoch)
        np.save(f"ssim_epoch_fold_{fold + 1}.npy", ssim_epoch)
        np.save(f"psnr_epoch_fold_{fold + 1}.npy", psnr_epoch)

    print("Training completed.")
