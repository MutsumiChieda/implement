import torch
import torch.nn as nn
import torch.nn.functional as F


class ATAC(nn.Module):
    """ ATAC unit. See Figure 1 in https://arxiv.org/pdf/2007.07729v1.pdf
    Args:
        channel_in (int): # of input channel
        r (float): channel reduction ratio (0 ~ 1)
    """

    def __init__(self, channel_in, r):
        super().__init__()
        channel_h = int(channel_in * r)
        self.point_conv1 = nn.Conv2d(
            channel_in,
            channel_h,
            kernel_size=1,
            groups=1,
            padding=0,
            dilation=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channel_h)
        self.relu = nn.ReLU()
        self.point_conv2 = nn.Conv2d(
            channel_h,
            channel_in,
            kernel_size=1,
            groups=1,
            padding=0,
            dilation=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channel_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.point_conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.point_conv2(h)
        h = self.bn2(h)
        h = self.sigmoid(h)

        return torch.mul(x, h)


class Block(nn.Module):
    def __init__(self, channel_in, channel_out, r=1):
        super().__init__()
        channel = channel_out // 4

        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.atac1 = ATAC(channel, r)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.atac2 = ATAC(channel, r)

        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)

        # Adjusting channels for skip connection
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.atac3 = ATAC(channel_out, r)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.atac1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.atac2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.atac3(h + shortcut)  # skip connection
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)


class GlobalAvgPool2d(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


class ResNet50ATAC(nn.Module):
    """ResNet50 but ReLU is replaced with ATAC module.
    Args:
        output_dim (int): # of output channels
        r (float): channel reduction ratio (0 ~ 1)
    """

    def __init__(self, output_dim, r=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        # Block 1
        self.block0 = self._building_block(256, channel_in=64, r=r)
        self.block1 = nn.ModuleList([self._building_block(256, r=r) for _ in range(2)])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        # Block 2
        self.block2 = nn.ModuleList([self._building_block(512, r=r) for _ in range(4)])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        # Block 3
        self.block3 = nn.ModuleList([self._building_block(1024, r=r) for _ in range(6)])
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        # Block 4
        self.block4 = nn.ModuleList([self._building_block(2048, r=r) for _ in range(3)])
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=-1)
        return y

    def _building_block(self, channel_out, channel_in=None, r=1):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out, r=r)
