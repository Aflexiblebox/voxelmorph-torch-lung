import torch
import numpy as np
from scipy import interpolate


def MSE(real_copy, predict_copy):
    # return mean_squared_error(real_copy, predict_copy)
    return torch.mean(torch.square(predict_copy - real_copy))


def calc_tre(disp_t2i, landmark_00_converted, landmark_disp, spacing):
    # x' = u(x) + x
    disp = np.array(disp_t2i.cpu())
    landmark_disp = np.array(landmark_disp.cpu())
    # convert -> z,y,x
    landmark_00_converted = np.array(landmark_00_converted[0].cpu())
    landmark_00_converted = np.flip(landmark_00_converted, axis=1)

    image_shape = disp.shape[1:]
    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(disp, 0, -1))
    calc_landmark_disp = inter(landmark_00_converted)

    diff = (np.sum(((calc_landmark_disp - landmark_disp) * spacing) ** 2, 1)) ** 0.5
    diff = diff[~np.isnan(diff)]

    return np.mean(diff), np.std(diff)
