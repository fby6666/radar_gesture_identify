import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RadarGestureDatasetTXT(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (string): 存储样本路径的 txt 文件
            transform (callable, optional): 预处理转换
        """
        self.transform = transform
        self.data_list = []

        # 读取 txt 文件
        with open(txt_file, "r") as f:
            for line in f.readlines():
                items = line.strip().split()  # 按空格拆分（如果是逗号分隔，使用 split(',')）
                if len(items) != 5:
                    continue  # 确保数据格式正确
                rt, dt, at_azimuth, at_elevation, label = items
                self.data_list.append((rt, dt, at_azimuth, at_elevation, int(label)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 读取图像
        rt_img = Image.open(self.data_list[idx][0]).convert("L")
        dt_img = Image.open(self.data_list[idx][1]).convert("L")
        at_azimuth_img = Image.open(self.data_list[idx][2]).convert("L")
        at_elevation_img = Image.open( self.data_list[idx][3]).convert("L")
        label=int(self.data_list[idx][4])
        # 预处理
        if self.transform:
            rt_img = self.transform(rt_img)
            dt_img = self.transform(dt_img)
            at_azimuth_img = self.transform(at_azimuth_img)
            at_elevation_img = self.transform(at_elevation_img)

        # 返回字典格式的样本
        sample = {
            "rt": rt_img,
            "dt": dt_img,
            "at_azimuth": at_azimuth_img,
            "at_elevation": at_elevation_img
        }
        return sample, label


