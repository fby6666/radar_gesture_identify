import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import torchvision.models as models


class MultiViewNet(nn.Module):
    def __init__(self, num_classes, share_weights=True):
        """
        Args:
            num_classes: 分类类别数
            share_weights: 是否使用共享的 ResNet18 提取器
        """
        super(MultiViewNet, self).__init__()

        # 加载预训练的 ResNet18，并去掉最后的全连接层
        base_resnet = models.resnet18(pretrained=True)
        base_resnet.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        # # 重新初始化权重（因为预训练参数是针对 RGB 图像的）
        # nn.init.kaiming_normal_(base_resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])  # 输出 (batch, 512, 1, 1)
        if share_weights:
            # 使用共享的特征提取器来处理所有视角
            self.extractor_front = self.feature_extractor
            self.extractor_back = self.feature_extractor
            self.extractor_left = self.feature_extractor
            self.extractor_right = self.feature_extractor
        else:
            # 如果不共享权重，则为每个视角单独构建一个分支（注意：这会增加参数量）
            self.extractor_front = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
            self.extractor_back = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
            self.extractor_left = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
            self.extractor_right = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])

        # 分类器：输入为 512*4 = 2048，输出 num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
            # 最终不用加 softmax，因为交叉熵损失会处理
        )

    def forward(self, inputs):
        """
        Args:
            inputs: 字典格式，包含四个视角的图像，如：
                {
                    "front": tensor,  # shape (batch, 1, H, W)
                    "back": tensor,
                    "left": tensor,
                    "right": tensor
                }
        Returns:
            logits: 预测的原始分数
        """
        # 分别提取各视角特征（输出形状：(batch, 512, 1, 1)）
        f_front = self.extractor_front(inputs["rt"])
        f_back = self.extractor_back(inputs["dt"])
        f_left = self.extractor_left(inputs["at_azimuth"])
        f_right = self.extractor_right(inputs["at_elevation"])

        # 展平特征向量：每个变为 (batch, 512)
        f_front = f_front.view(f_front.size(0), -1)
        f_back = f_back.view(f_back.size(0), -1)
        f_left = f_left.view(f_left.size(0), -1)
        f_right = f_right.view(f_right.size(0), -1)

        # 拼接四个特征向量，得到 (batch, 2048)
        f_cat = torch.cat([f_front, f_back, f_left, f_right], dim=1)

        # 分类得到 logits
        logits = self.classifier(f_cat)
        return logits

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Dataset import *

def dataset_loader(txt_file,batch_size=32,ratio=0.8,shuffle=True, transform=transforms):
    # 图像进行预处理，同时将数据集输入dataloader
    # 创建数据集实例
    # txt_file = './data_list.txt'  # txt 文件路径
    dataset = RadarGestureDatasetTXT(txt_file=txt_file, transform=transform)
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

# if __name__ == '__main__':
#     trans= transforms.Compose([
#         transforms.Resize((224, 224)),  # 统一尺寸
#         transforms.ToTensor(),  # 转换为 PyTorch Tensor
#         # transforms.Normalize(mean=[...], std=[...])  # 归一化（如需）
#     ])
#     tarin_iter,test_iter=dataset_loader('./data_list.txt',transform=trans)
#     # 测试一个 batch
#     for batch_samples, batch_labels in tarin_iter:
#         # 确保 batch size 只有 1 以便 debug
#         net = MultiViewNet(num_classes=4)
#         # 将 4 张灰度图作为输入，确保输入维度匹配
#         output = net(batch_samples)  # 假设 net 接受 4 个输入
#         print("网络输出:", output.shape)  # 期望输出: torch.Size([batch_size, num_classes])
#         break


