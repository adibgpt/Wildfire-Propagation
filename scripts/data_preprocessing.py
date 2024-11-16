# Script for data preprocessing and normalization

import numpy as np
import pickle
import torch
import math

# Helper function to normalize data
def normalize(data):
    """
    Normalize the data to a range of [0, 1].

    Args:
        data (numpy.ndarray): The input data to normalize.

    Returns:
        tuple: Normalized data, minimum value, and maximum value.
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    return (data - data_min) / (data_max - data_min), data_min, data_max

# Convert degrees to radians
def degrees_to_radians(degrees):
    """
    Convert degrees to radians.

    Args:
        degrees (float): Angle in degrees.

    Returns:
        float: Angle in radians.
    """
    return degrees * (math.pi / 180)

# Main preprocessing function
def preproc_one_dataset(infile, zag_ind, ramp, u10, x_extract_every_array, y_extract_every_array, ustar_or_u10, power_of_two):
    """
    Preprocess a single dataset.

    Args:
        infile (str): Path to the input file.
        zag_ind (int): Index of the zag to use as inputs.
        ramp (float): Ramp value for terrain adjustment.
        u10 (float): Wind speed at 10 meters height.
        x_extract_every_array (int): Sampling step in x direction.
        y_extract_every_array (int): Sampling step in y direction.
        ustar_or_u10 (str): Indicator for velocity type ('ustar' or 'u10').
        power_of_two (bool): Whether to pad the data to a power of two in size.

    Returns:
        tuple: Processed data and associated metadata.
    """
    with open(infile, 'rb') as file:
        filt_data, xi_f_filt, x_filt, y_filt = pickle.load(file)

    theta_terrain_filt = filt_data['theta']
    if ustar_or_u10 == 'ustar':
        u_terrain_filt = (
            filt_data['u'] * np.cos(degrees_to_radians(ramp)) +
            filt_data['w'] * np.sin(degrees_to_radians(ramp))
        )
    elif ustar_or_u10 == 'u10':
        u_terrain_filt = np.ones(np.shape(theta_terrain_filt)) * u10

    x_filt = np.arange(np.shape(theta_terrain_filt)[1]) * x_extract_every_array + 100
    y_filt = np.arange(np.shape(theta_terrain_filt)[2]) * y_extract_every_array

    xi_f_filt = np.maximum(xi_f_filt, 300)

    theta_terrain_filt = theta_terrain_filt[:, :, :, zag_ind]
    u_terrain_filt = u_terrain_filt[:, :, :, zag_ind]

    nt, ny = xi_f_filt.shape
    nx = int(np.shape(theta_terrain_filt)[1])

    xi_f_mat = np.zeros((nt, nx, ny))
    x_loc = np.arange(nx) * x_extract_every_array + 100
    y_loc = np.arange(ny) * y_extract_every_array

    for t in range(nt):
        dist_x = np.abs(x_loc[:, np.newaxis] - xi_f_filt[t])
        sign_dist_x = np.sign(x_loc[:, np.newaxis] - xi_f_filt[t])
        for y in range(ny):
            dist_y = np.abs(y_loc - y * y_extract_every_array)
            dist_matrix = np.sqrt(dist_x ** 2 + dist_y ** 2)
            argmin_indices = np.argmin(dist_matrix, axis=1)
            selected_sign_dist_x = sign_dist_x[np.arange(dist_matrix.shape[0]), argmin_indices]
            xi_f_mat[t, :, y] = np.min(dist_matrix, axis=1) * selected_sign_dist_x

    theta, theta_min, theta_max = normalize(theta_terrain_filt)
    if ustar_or_u10 == 'ustar':
        u_in, u_min, u_max = normalize(u_terrain_filt)
    elif ustar_or_u10 == 'u10':
        u_min = 2
        u_max = 10
        u_in = (u_terrain_filt - u_min) / (u_max - u_min)
    dist_xi, dist_xi_min, dist_xi_max = normalize(xi_f_mat)
    xi_f_normalized, xi_f_min, xi_f_max = normalize(xi_f_filt)

    ramp_mat = np.ones(np.shape(theta)) * ramp
    ramp_mat = (ramp_mat - 0) / (30 - 0)

    theta = torch.from_numpy(theta)
    dist_xi = torch.from_numpy(dist_xi)
    u_in = torch.from_numpy(u_in)
    ramp_mat = torch.from_numpy(ramp_mat)

    data = torch.stack([theta, dist_xi, u_in, ramp_mat], dim=1)
    data = data[1:]
    xi_f_target = xi_f_normalized[1:]
    data = data.float()

    if power_of_two and x_extract_every_array == 8:
        temp_B, temp_C, H_old, temp_W = data.shape
        H_new = 128
        temp_data = torch.zeros(temp_B, temp_C, H_new, temp_W)
        temp_data[:, :, :H_old, :] = data
        temp_data[:, :, H_old:, :] = data[:, :, -1:, :].expand(-1, -1, H_new - H_old, -1)
        data = temp_data
        for _ in range(H_new - H_old):
            next_value = x_filt[-1] + (x_filt[-1] - x_filt[-2])
            x_filt = np.append(x_filt, next_value)

    nt, n_channels, nx, ny = data.shape

    return (theta_terrain_filt, u_terrain_filt, xi_f_filt, xi_f_mat, x_filt, y_filt,
            theta_min, theta_max, u_min, u_max, dist_xi_min, dist_xi_max, xi_f_min,
            xi_f_max, data, xi_f_target, nt, n_channels, nx, ny)
