# Script for visualizing training results

import matplotlib.pyplot as plt
import numpy as np

def plot_metric(metric_values, metric_name, fold=None, save_path=None):
    """
    Plot a given metric over epochs.

    Args:
        metric_values (list or numpy array): Metric values for each epoch.
        metric_name (str): Name of the metric (e.g., "MSE Loss", "SSIM").
        fold (int, optional): Fold number for the plot title. Default is None.
        save_path (str, optional): Path to save the plot. Default is None.
    """
    plt.figure(dpi=200)
    plt.plot(metric_values, label=f"Average {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Epochs" + (f" (Fold {fold})" if fold is not None else ""))
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def visualize_results(loss_files, ssim_files, psnr_files, save_dir="plots"):
    """
    Visualize and save training metrics for all folds.

    Args:
        loss_files (list of str): List of file paths for loss values (per fold).
        ssim_files (list of str): List of file paths for SSIM values (per fold).
        psnr_files (list of str): List of file paths for PSNR values (per fold).
        save_dir (str): Directory to save plots.
    """
    avg_loss = []
    avg_ssim = []
    avg_psnr = []

    # Iterate over folds
    for i, (loss_file, ssim_file, psnr_file) in enumerate(zip(loss_files, ssim_files, psnr_files)):
        # Load metrics
        loss_values = np.load(loss_file)
        ssim_values = np.load(ssim_file)
        psnr_values = np.load(psnr_file)

        avg_loss.append(loss_values)
        avg_ssim.append(ssim_values)
        avg_psnr.append(psnr_values)

        # Plot metrics for the current fold
        plot_metric(loss_values, "MSE Loss", fold=i + 1, save_path=f"{save_dir}/mse_loss_fold_{i + 1}.png")
        plot_metric(ssim_values, "SSIM", fold=i + 1, save_path=f"{save_dir}/ssim_fold_{i + 1}.png")
        plot_metric(psnr_values, "PSNR (dB)", fold=i + 1, save_path=f"{save_dir}/psnr_fold_{i + 1}.png")

    # Calculate and plot average metrics across folds
    avg_loss_epoch = np.mean(avg_loss, axis=0)
    avg_ssim_epoch = np.mean(avg_ssim, axis=0)
    avg_psnr_epoch = np.mean(avg_psnr, axis=0)

    plot_metric(avg_loss_epoch, "Average MSE Loss", save_path=f"{save_dir}/avg_mse_loss.png")
    plot_metric(avg_ssim_epoch, "Average SSIM", save_path=f"{save_dir}/avg_ssim.png")
    plot_metric(avg_psnr_epoch, "Average PSNR (dB)", save_path=f"{save_dir}/avg_psnr.png")

    print("Visualization completed.")

if __name__ == "__main__":
    # Example usage
    loss_files = [f"loss_epoch_fold_{i + 1}.npy" for i in range(7)]
    ssim_files = [f"ssim_epoch_fold_{i + 1}.npy" for i in range(7)]
    psnr_files = [f"psnr_epoch_fold_{i + 1}.npy" for i in range(7)]

    visualize_results(loss_files, ssim_files, psnr_files, save_dir="plots")
