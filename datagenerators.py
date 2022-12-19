import platform

import numpy as np
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import torch.nn.functional as F


def data_standardization_0_n(range, img):
    if torch.is_tensor(img):
        return range * (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    else:
        return range * (img - np.min(img)) / (np.max(img) - np.min(img))


class Dataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files):
        self.moving_files = moving_files
        self.fixed_files = fixed_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        # shape dosen't match
        if m_img.shape != f_img.shape:
            img_tensor = F.interpolate(torch.tensor(m_img).unsqueeze(0), size=f_img.shape[1:],
                                       mode='trilinear',
                                       align_corners=False)

            m_img = np.array(img_tensor)[0, ...]

        # if self.moving_files[index].split('moving/')[1] != self.fixed_files[index].split('fixed/')[1]:
        #     print("=================================================")
        #     print("{} is not match {}".format(self.moving_files[index].split('moving/')[1],
        #                                       self.fixed_files[index].split('fixed/')[1]))
        #     print("=================================================")
        # raise ValueError

        return [m_img, self.moving_files[index]], [f_img, self.fixed_files[index]]


class TestDataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files, landmark_files):
        # 初始化
        self.moving_files = moving_files
        self.fixed_files = fixed_files
        self.landmark_files = landmark_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        m_name = self.moving_files[index].split('moving\\')[1] if platform.system().lower() == 'windows' else \
        self.moving_files[index].split('moving/')[1]
        f_name = self.fixed_files[index].split('fixed\\')[1] if platform.system().lower() == 'windows' else \
        self.fixed_files[index].split('fixed/')[1]

        return [m_img, m_name], [f_img, f_name], self.landmark_files[index]
