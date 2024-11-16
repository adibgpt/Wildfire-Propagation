import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


class STE_Loss(torch.nn.Module):
    """Custom loss to compute the sum difference error (STE)."""
    def forward(self, prediction, target):
        return torch.abs(torch.sum(prediction) - torch.sum(target)) / prediction.numel()


class JaccardSimilarity(torch.nn.Module):
    """Custom IoU (Jaccard Similarity) for flame thresholded data."""
    def forward(self, pred, target, threshold=0.1):
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()
        intersection = torch.sum(pred_bin * target_bin)
        union = torch.sum(pred_bin) + torch.sum(target_bin) - intersection
        return intersection / union


# Initialize metrics
calc_MSE_Loss = torch.nn.MSELoss()
calc_STE_Loss = STE_Loss()
calc_Jaccard = JaccardSimilarity()


def evaluate_fireline_predictions(
    model, 
    data, 
    xi_f_filt, 
    x_filt, 
    y_filt, 
    theta_min, 
    theta_max, 
    u_min, 
    u_max, 
    dist_xi_min, 
    dist_xi_max, 
    save_prefix, 
    sequence_length, 
    device, 
    n_gpu=1
):
    """
    Evaluate fireline predictions and visualize results including animations.

    Args:
        model (torch.nn.Module): Trained model.
        data (torch.Tensor): Test data tensor.
        xi_f_filt (np.ndarray): Ground truth fire line positions.
        x_filt (np.ndarray): Spatial grid x-coordinates.
        y_filt (np.ndarray): Spatial grid y-coordinates.
        theta_min, theta_max, u_min, u_max, dist_xi_min, dist_xi_max (float): Normalization parameters.
        save_prefix (str): File name prefix for saving outputs.
        sequence_length (int): Sequence length for model input.
        device (torch.device): Computation device.
        n_gpu (int): Number of GPUs in use.
    """
    model.eval()
    torch.cuda.empty_cache()

    current_data_sequence = data[:sequence_length].unsqueeze(0).to(device)
    image_predictions = []
    xi_f_predictions = []
    n_predictions = len(data) - sequence_length

    with torch.no_grad():
        for step in range(n_predictions):
            # Initialize hidden states
            hidden = model.module.convlstm.init_hidden(current_data_sequence.size(0)) if n_gpu > 1 else model.convlstm.init_hidden(current_data_sequence.size(0))
            
            # Predict next step
            next_step = model(current_data_sequence, hidden)
            image_predictions.append(next_step.squeeze(0).cpu().numpy())
            
            # Fireline prediction (dist_xi)
            dist_xi_pred = next_step[:, 1, :, :].squeeze(0).cpu().numpy()
            xi_f_pred = np.argmax(dist_xi_pred, axis=0) * (data.shape[-1] // dist_xi_pred.shape[0])
            xi_f_predictions.append(xi_f_pred)
            
            # Update sequence
            current_data_sequence = torch.cat((current_data_sequence[:, 1:], next_step.unsqueeze(1).to(device)), dim=1)

    # Convert predictions to tensors
    image_predictions = torch.tensor(np.array(image_predictions))
    xi_f_predictions = np.array(xi_f_predictions)

    # Calculate and save metrics
    eval_metrics = calculate_fireline_metrics(image_predictions, data[sequence_length:], xi_f_filt, xi_f_predictions)
    with open(f"{save_prefix}_metrics.pkl", "wb") as f:
        pickle.dump(eval_metrics, f)

    # Generate visualizations
    generate_fireline_visualizations(image_predictions, data, xi_f_predictions, xi_f_filt, x_filt, y_filt, dist_xi_min, dist_xi_max, save_prefix)


def calculate_fireline_metrics(predictions, ground_truth, xi_f_filt, xi_f_predictions):
    """
    Compute metrics such as MSE, STE, and Jaccard Similarity for fireline predictions.

    Args:
        predictions (torch.Tensor): Predicted data.
        ground_truth (torch.Tensor): Ground truth data.
        xi_f_filt (np.ndarray): Ground truth fireline positions.
        xi_f_predictions (np.ndarray): Predicted fireline positions.

    Returns:
        dict: Dictionary of calculated metrics.
    """
    metrics = {
        "MSE_all": [],
        "STE_all": [],
        "Jaccard_all": []
    }

    for i in range(len(predictions)):
        pred = predictions[i]
        target = ground_truth[i]

        metrics["MSE_all"].append(calc_MSE_Loss(pred, target).item())
        metrics["STE_all"].append(calc_STE_Loss(pred, target).item())
        metrics["Jaccard_all"].append(calc_Jaccard(pred, target).item())

    return metrics


def generate_fireline_visualizations(predictions, data, xi_f_predictions, xi_f_filt, x_filt, y_filt, dist_xi_min, dist_xi_max, save_prefix):
    """
    Generate and save visualizations including animations for fireline differences.

    Args:
        predictions (torch.Tensor): Predicted data.
        data (torch.Tensor): Ground truth data.
        xi_f_predictions (np.ndarray): Predicted fireline positions.
        xi_f_filt (np.ndarray): Ground truth fireline positions.
        x_filt, y_filt (np.ndarray): Spatial grid coordinates.
        dist_xi_min, dist_xi_max (float): Normalization parameters for dist_xi.
        save_prefix (str): File name prefix for saving visualizations.
    """
    # Animation for fireline predictions
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    data_np = data.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    cax = ax.pcolor(
        x_filt, y_filt, 
        predictions_np[0, 1, :, :].T, 
        vmin=dist_xi_min, vmax=dist_xi_max, shading='auto', cmap='cividis'
    )
    cbar = fig.colorbar(cax, orientation="horizontal")
    ground_truth_line, = ax.plot(xi_f_filt[0, :], y_filt, color='white', linewidth=2, label="Ground Truth Fireline")
    prediction_line, = ax.plot(xi_f_predictions[0, :], y_filt, color='red', linewidth=2, label="Prediction Fireline")
    ax.set_aspect("equal")
    ax.set_title("Fireline Prediction and Ground Truth")

    def update_fireline(frame):
        cax.set_array(predictions_np[frame, 1, :, :].T.ravel())
        ground_truth_line.set_xdata(xi_f_filt[frame, :])
        prediction_line.set_xdata(xi_f_predictions[frame, :])
        ax.set_title(f"Fireline Prediction at Time Step {frame}")
        return cax, ground_truth_line, prediction_line

    ani = animation.FuncAnimation(fig, update_fireline, frames=len(predictions), blit=True, interval=200)
    ani.save(f"{save_prefix}_fireline_prediction.gif", dpi=200)

    print(f"Visualizations saved with prefix {save_prefix}.")


# Example usage (adapt parameters as needed)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ...  # Load trained model
    data = ...  # Load test data
    xi_f_filt, x_filt, y_filt = ...  # Load spatial data
    dist_xi_min, dist_xi_max = ...  # Normalization parameters

    evaluate_fireline_predictions(
        model=model,
        data=data,
        xi_f_filt=xi_f_filt,
        x_filt=x_filt,
        y_filt=y_filt,
        theta_min=None,
        theta_max=None,
        u_min=None,
        u_max=None,
        dist_xi_min=dist_xi_min,
        dist_xi_max=dist_xi_max,
        save_prefix="results/fireline_eval",
        sequence_length=5,
        device=device
    )
