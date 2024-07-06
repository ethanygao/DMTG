""" 
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from .ozan_rep_fun import ozan_rep_function,trevor_rep_function,OzanRepFunction,TrevorRepFunction
from .models import register

# __all__ = ['xception_taskonomy_new','xception_taskonomy_new_fifth','xception_taskonomy_new_quad','xception_taskonomy_new_half','xception_taskonomy_new_80','xception_taskonomy_ozan']

# model_urls = {
#     'xception_taskonomy':'file:///home/tstand/Dropbox/taskonomy/xception_taskonomy-a4b32ef7.pth.tar'
# }
SIZES = {1: (32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728),
         0.2: (16, 32, 64, 256, 320, 320, 320, 320, 320, 320, 320, 320, 320),
         0.3: (32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728),
         0.4: (32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728),
         0.5: (24, 48, 96, 192, 512, 512, 512, 512, 512, 512, 512, 512, 512),
         0.8: (32, 64, 128, 248, 648, 648, 648, 648, 648, 648, 648, 648, 648),
         2: (32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728),
         4: (64, 128, 256, 512, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456)}

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,groupsize=1):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=max(1,in_channels//groupsize),bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        #self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
        #self.pointwise=lambda x:x
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters=out_filters

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            #rep.append(nn.AvgPool2d(3,strides,1))
            rep.append(nn.Conv2d(filters,filters,2,2))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x+=skip
        return x


class XceptionEncoder(nn.Module):
    def __init__(self, sizes):
        super(XceptionEncoder, self).__init__()
        assert len(sizes) >=2, "Length of sizes here must be greater than 2"
        self.conv1 = nn.Conv2d(3, sizes[0], 3,2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(sizes[0],sizes[1],3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(sizes[1])
        #do relu here

        self.blocks = nn.ModuleList()
        for i, l in enumerate(range(2, len(sizes))):
            if sizes[l-1] != sizes[l]:
                if i == 0:
                    self.blocks.append(Block(sizes[l-1], sizes[l], 2, 2, start_with_relu=False, grow_first=True))
                else:
                    self.blocks.append(Block(sizes[l-1], sizes[l], 2, 2, start_with_relu=True, grow_first=True))
            else:
                self.blocks.append(Block(sizes[l-1], sizes[l], 3, 1, start_with_relu=True, grow_first=True))

    def forward(self,input):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        for block in self.blocks:
            x = block(x)
        return x


class XceptionGroupEncoder(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, sizes, relu=True):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionGroupEncoder, self).__init__()
        assert len(sizes) >= 1, "Length of sizes here must be greater than 2"
        self.blocks = nn.ModuleList()
        for i, l in enumerate(range(1, len(sizes))):
            if sizes[l-1] != sizes[l]:
                if i == 0:
                    self.blocks.append(Block(sizes[l-1], sizes[l], 2, 2, start_with_relu=relu, grow_first=True))
                else:
                    self.blocks.append(Block(sizes[l - 1], sizes[l], 2, 2, start_with_relu=True, grow_first=True))
            else:
                self.blocks.append(Block(sizes[l-1], sizes[l], 3, 1, start_with_relu=True, grow_first=True))

        pre_rep_size = sizes[-1]
        self.relu = nn.ReLU()
        self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
        self.final_conv_bn = nn.BatchNorm2d(512)

    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x)
        rep = self.relu(x)
        rep = self.final_conv(rep)
        rep = self.final_conv_bn(rep)

        return rep


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


@register("xception_encoder")
def build_xception_encoder(param_set_num, first_k_layers):
    sizes_ = SIZES[param_set_num][:first_k_layers]
    encoder_ = XceptionEncoder(sizes_)
    encoder_.apply(init_weights)
    return encoder_


@register("xception_group_encoder")
def build_xception_group_encoder(param_set_num, behind_first_k_layers, relu=True):
    sizes_ = SIZES[param_set_num][behind_first_k_layers-1:]
    relu_ = relu
    encoder_ = XceptionGroupEncoder(sizes_, relu_)
    encoder_.apply(init_weights)
    return encoder_


@register("xception")
def build_xception(param_set_num, split_point):
    sizes_ = SIZES[param_set_num]
    encoders = []
    for i in range(2):
        if i == 0:
            encoder = XceptionEncoder(sizes_[:split_point])
        else:
            encoder = XceptionGroupEncoder(sizes_[split_point-1:])
        encoder.apply(init_weights)
        encoders.append(encoder)
    return nn.Sequential(*encoders)


