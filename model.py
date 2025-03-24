import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import torchvision.models as models


class MultiViewNet(nn.Module):
    def __init__(self, num_classes, share_weights=False):
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
            # 每个视角单独创建新的 ResNet18
            self.extractor_front = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.extractor_front.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.extractor_front = nn.Sequential(*list(self.extractor_front.children())[:-1])

            self.extractor_back = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.extractor_back.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.extractor_back = nn.Sequential(*list(self.extractor_back.children())[:-1])

            self.extractor_left = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.extractor_left.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.extractor_left = nn.Sequential(*list(self.extractor_left.children())[:-1])

            self.extractor_right = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.extractor_right.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.extractor_right = nn.Sequential(*list(self.extractor_right.children())[:-1])

        # 分类器：输入为 512*4 = 2048，输出 num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 512*4),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
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



