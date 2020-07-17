import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class UNet(nn.Module):
    """
    UNet model
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv_1 = DoubleConv(in_channels, 64)
        self.conv_2 = DoubleConv(64, 128)
        self.conv_3 = DoubleConv(128, 256)
        self.conv_4 = DoubleConv(256, 512)
        self.conv_5 = DoubleConv(512, 1024)
        self.down = nn.MaxPool2d(2)
        self.cat_down = Cat()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_6 = DoubleConv(1024, 512)
        self.conv_7 = DoubleConv(512, 256)
        self.conv_8 = DoubleConv(256, 128)
        self.conv_9 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(self.down(x1))
        x3 = self.conv_3(self.down(x2))
        x4 = self.conv_4(self.down(x3))
        x = self.conv_5(self.down(x4))
        x = self.conv_6(self.cat_down(self.up(x), x4))
        x = self.conv_7(self.cat_down(self.up(x), x3))
        x = self.conv_8(self.cat_down(self.up(x), x2))
        x = self.conv_9(self.cat_down(self.up(x), x1))
        x = self.out_conv(x)
        return x


class DoubleConv(nn.Module):
    """
    Double 3x3 conv + relu
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.Relu()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        return x


class Cat(nn.Module):
    """
    Concat expansive tensor with contracting tensor
    need crop the contracting tensor (use conv2d)
    """

    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, up, down):
        ch = down.size()[1]
        dw = down.size()[2] - up.size()[2]
        dh = down.size()[3] - up.size()[3]
        down = nn.Conv2d(ch, ch, (dw + 1, dh + 1))(down)
        y = torch.cat([down, up], dim=1)
        return y
