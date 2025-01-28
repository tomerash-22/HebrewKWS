import torch
import torch.nn as nn
import torch.nn.functional as F


# Regular convolution block: Conv -> BatchNorm -> ReLU
class RegularConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(RegularConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding="same",
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Depthwise separable convolution block: Depthwise Conv -> BatchNorm -> ReLU -> Pointwise Conv -> BatchNorm -> ReLU
class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DSConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding="same",
                                   groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


# DS-CNN model
class DSCNN(nn.Module):
    def __init__(self, input_shape=(1, 10, 50), output_size=64):
        super(DSCNN, self).__init__()
        in_channels = input_shape[0]

        # Regular convolution block
        self.regular_conv = RegularConvBlock(in_channels, 16, kernel_size=(10, 3))

        # DSConv blocks
        self.dsconv1 = DSConvBlock(16, 64, kernel_size=(10, 3))
        self.dsconv2 = DSConvBlock(64, 64, kernel_size=(10, 3))

        # Global average pooling (flattened for feature representation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.regular_conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)

        # Global average pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation for the output layer
        return x


