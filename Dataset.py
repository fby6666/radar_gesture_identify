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
                    print(f"Invalid line in txt file: {line}")
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

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Dataset import *

def dataset_loader(txt_file,batch_size=32,ratio=0.8,shuffle=True, transform=transforms):
    # 图像进行预处理，同时将数据集输入dataloader
    # 创建数据集实例
    # txt_file = './data_list.txt'  # txt 文件路径
    dataset = RadarGestureDatasetTXT(txt_file=txt_file, transform=transform)
    print(type(dataset))
    train_size=int(len(dataset)*ratio)
    test_size = len(dataset) - train_size
    print(f'train_samples_num: {train_size},test_samples_num: {test_size}')
    # 划分数据集
    generator1 = torch.Generator().manual_seed(42)
    train_dataset,test_dataset=torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator1)
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader,test_loader
