# Post-training predictions and visualizations

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Custom loss definitions
class STE_Loss(torch.nn.Module):
    def forward(self, prediction, target):
        return torch.abs(torch.sum(prediction) - torch.sum(target)) / prediction.numel()

class JaccardSimilarity(torch.nn.Module):
    def forward(self, pred, target, threshold=0.1):
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()
        intersection = torch.sum(pred_bin * target_bin)
        union = torch.sum(pred_bin) + torch.sum(target_bin) - intersection
        return intersection / union

# Prediction and evaluation
def evaluate_predictions(model, data, xi_f_filt, device, metrics, save_prefix, sequence_length=5, n_gpu=1):
    """
    Evaluate the model predictions and calculate metrics.

    Args:
        model (torch.nn.Module): Trained model.
        data (torch.Tensor): Input data tensor.
        xi_f_filt (np.ndarray): Ground truth fire line positions.
        device (torch.device): Device for computation.
        metrics (dict): Dictionary of metric functions.
        save_prefix (str): Prefix for saving plots and metrics.
        sequence_length (int): Sequence length for input data.
        n_gpu (int): Number of GPUs used.
    """
    model.eval()
    torch.cuda.empty_cache()

    # Prepare the initial sequence
    current_data_sequence = data[:sequence_length].unsqueeze(0).to(device)
    predictions = []
    xi_f_predictions = []
    n_predictions = len(data) - sequence_length

    with torch.no_grad():
        for step in range(n_predictions):
            # Initialize hidden state
            hidden = (model.module.init_hidden(current_data_sequence.size(0)) 
                      if n_gpu > 1 else model.init_hidden(current_data_sequence.size(0)))
            
            # Predict the next step
            output = model(current_data_sequence, hidden)
            predictions.append(output.squeeze(0).cpu().numpy())
            
            # Extract fire line positions
            dist_xi_pred = output[:, 1, :, :].squeeze(0).cpu().numpy()
            xi_f_pred = np.argmax(dist_xi_pred, axis=0) * (data.shape[-1] // dist_xi_pred.shape[0])
            xi_f_predictions.append(xi_f_pred)

            # Update the input sequence
            current_data_sequence = torch.cat((current_data_sequence[:, 1:], output.unsqueeze(1).to(device)), dim=1)

    # Convert predictions to tensors
    predictions = torch.tensor(np.array(predictions))
    xi_f_predictions = np.array(xi_f_predictions)

    # Calculate metrics
    eval_metrics = {name: [] for name in metrics.keys()}
    for i in range(n_predictions):
        pred = predictions[i]
        target = data[sequence_length + i].to(device)

        for name, metric in metrics.items():
            eval_metrics[name].append(metric(pred, target).item())

    # Save metrics
    with open(f"{save_prefix}_metrics.pkl", "wb") as f:
        pickle.dump(eval_metrics, f)

    # Visualize results
    visualize_predictions(predictions, xi_f_predictions, data, xi_f_filt, save_prefix, eval_metrics)

def visualize_predictions(predictions, xi_f_predictions, data, xi_f_filt, save_prefix, metrics):
    """
    Generate and save visualizations for predictions.

    Args:
        predictions (torch.Tensor): Predicted data.
        xi_f_predictions (np.ndarray): Predicted fire line positions.
        data (torch.Tensor): Ground truth data.
        xi_f_filt (np.ndarray): Ground truth fire line positions.
        save_prefix (str): Prefix for saving visualizations.
        metrics (dict): Metrics for plotting.
    """
    # Plot metrics
    for metric_name, values in metrics.items():
        plt.figure(dpi=200)
        plt.plot(values, label=f"{metric_name}")
        plt.xlabel("Time Step")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_prefix}_{metric_name.lower()}.png", dpi=200)

    # Animation for predictions
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    im = ax.imshow(predictions[0, 1].cpu(), cmap="cividis", aspect="auto")
    true_line, = ax.plot(xi_f_filt[0], np.arange(len(xi_f_filt[0])), "w-", label="True Fire Line")
    pred_line, = ax.plot(xi_f_predictions[0], np.arange(len(xi_f_predictions[0])), "r-", label="Predicted Fire Line")

    def update(frame):
        im.set_data(predictions[frame, 1].cpu())
        true_line.set_xdata(xi_f_filt[frame])
        pred_line.set_xdata(xi_f_predictions[frame])
        ax.set_title(f"Frame {frame}")
        return im, true_line, pred_line

    ani = animation.FuncAnimation(fig, update, frames=len(predictions), interval=200, blit=True)
    ani.save(f"{save_prefix}_fireline_animation.gif", dpi=200)

    print(f"Visualizations saved with prefix {save_prefix}.")

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ...  # Load trained model
    data = ...  # Load test data
    xi_f_filt = ...  # Ground truth fire line positions

    metrics = {
        "MSE": torch.nn.MSELoss(),
        "STE": STE_Loss(),
        "Jaccard": JaccardSimilarity(),
        "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        "PSNR": PeakSignalNoiseRatio(data_range=1.0).to(device),
    }

    evaluate_predictions(model, data, xi_f_filt, device, metrics, save_prefix="results/fireline")
