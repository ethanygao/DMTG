import torch
from torch import nn
from torch.nn import functional as F
from .models import register


class ResBlock(nn.Module):

    def __init__(self, filters, kernel_size, strides):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_size[0], stride=strides, padding=kernel_size[0]//2, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[1])

        self.conv2 = nn.Conv2d(filters[1], filters[2], kernel_size=kernel_size[1], stride=1, padding=kernel_size[1]//2, bias=False)
        self.bn2 = nn.BatchNorm2d(filters[2])

        if strides == (1, 1):
            self.shortcut = nn.Identity()
        else:
            shortcut_ = []
            shortcut_.append(nn.Conv2d(filters[0], filters[2], kernel_size=1, stride=2, bias=False))
            shortcut_.append(nn.BatchNorm2d(filters[2]))
            self.shortcut = nn.Sequential(*shortcut_)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + self.shortcut(inputs)
        return F.relu(x)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(3, 2)

        self.resblock_2 = ResBlock((64, 64, 64), (3, 3), (1, 1))
        self.avg_pooling = nn.AvgPool2d(2)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.max_pooling(x)
        x = self.resblock_2(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ResNet18Minus(nn.Module):
    def __init__(self):
        super(ResNet18Minus, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(3, 2)

        # self.resblock_2 = ResBlock((64, 64, 64), (3, 3), (1, 1))
        # self.avg_pooling = nn.AvgPool2d(2)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.max_pooling(x)
        # x = self.resblock_2(x)
        # x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ResNet18Base(nn.Module):
    def __init__(self):
        super(ResNet18Base, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(3, 2)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.max_pooling(x)
        return x


class ResNet18Group(nn.Module):
    def __init__(self):
        super(ResNet18Group, self).__init__()
        self.resblock_2 = ResBlock((64, 64, 64), (3, 3), (1, 1))
        self.avg_pooling = nn.AvgPool2d(2)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, inputs):
        x = self.resblock_2(inputs)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        return x


@register("resnet_base")
def build_resnet_base(**kwargs):
    return ResNet18Base()


@register("resnet_group")
def build_resnet_group(**kwargs):
    return ResNet18Group()


@register("simple_resnet")
def build_simple_resnet(**kwargs):
    return ResNet18()


@register("simple_resnet_minus")
def build_simple_resnet(**kwargs):
    return ResNet18Minus()

