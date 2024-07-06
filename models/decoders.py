import torch
from torch import nn
from torch.nn import functional as F
from .models import register
from .xception_taskonomy_grouping import Block, SeparableConv2d


class AttributeDecoder(nn.Module):
    def __init__(self, inp_feat, n_classes, prob):
        super(AttributeDecoder, self).__init__()
        self.fc1 = nn.Linear(inp_feat, n_classes)
        self.prob = prob

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.apply(init_weights)

    def forward(self, inputs):
        x = F.dropout1d(inputs, p=self.prob, training=self.training)
        x = self.fc1(x)
        return x


class AttributeDecoderNoDrop(AttributeDecoder):
    def __init__(self, inp_feat, n_classes):
        super(AttributeDecoderNoDrop, self).__init__(inp_feat, n_classes, 0.5)

    def forward(self, inputs):
        x = inputs
        x = self.fc1(x)
        return x


class ImgDecoder(nn.Module):
    def __init__(self, out_channel=32, num_classes=None, half_sized_output=False, small_decoder=True):
        super(ImgDecoder, self).__init__()

        self.output_channels = out_channel
        self.num_classes = num_classes
        self.half_sized_output = half_sized_output
        self.relu = nn.ReLU(inplace=True)
        if num_classes is not None:
            self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

            self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(1536)

            # do relu here
            self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
            self.bn4 = nn.BatchNorm2d(2048)

            self.fc = nn.Linear(2048, num_classes)
        else:
            if small_decoder:
                self.upconv1 = nn.ConvTranspose2d(512, 128, 2, 2)
                self.bn_upconv1 = nn.BatchNorm2d(128)
                self.conv_decode1 = nn.Conv2d(128, 128, 3, padding=1)
                self.bn_decode1 = nn.BatchNorm2d(128)
                self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
                self.bn_upconv2 = nn.BatchNorm2d(64)
                self.conv_decode2 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn_decode2 = nn.BatchNorm2d(64)
                self.upconv3 = nn.ConvTranspose2d(64, 48, 2, 2)
                self.bn_upconv3 = nn.BatchNorm2d(48)
                self.conv_decode3 = nn.Conv2d(48, 48, 3, padding=1)
                self.bn_decode3 = nn.BatchNorm2d(48)
                if half_sized_output:
                    self.upconv4 = nn.Identity()
                    self.bn_upconv4 = nn.Identity()
                    self.conv_decode4 = nn.Conv2d(48, out_channel, 3, padding=1)
                else:
                    self.upconv4 = nn.ConvTranspose2d(48, 32, 2, 2)
                    self.bn_upconv4 = nn.BatchNorm2d(32)
                    self.conv_decode4 = nn.Conv2d(32, out_channel, 3, padding=1)
            else:
                self.upconv1 = nn.ConvTranspose2d(512, 256, 2, 2)
                self.bn_upconv1 = nn.BatchNorm2d(256)
                self.conv_decode1 = nn.Conv2d(256, 256, 3, padding=1)
                self.bn_decode1 = nn.BatchNorm2d(256)
                self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
                self.bn_upconv2 = nn.BatchNorm2d(128)
                self.conv_decode2 = nn.Conv2d(128, 128, 3, padding=1)
                self.bn_decode2 = nn.BatchNorm2d(128)
                self.upconv3 = nn.ConvTranspose2d(128, 96, 2, 2)
                self.bn_upconv3 = nn.BatchNorm2d(96)
                self.conv_decode3 = nn.Conv2d(96, 96, 3, padding=1)
                self.bn_decode3 = nn.BatchNorm2d(96)
                if half_sized_output:
                    self.upconv4 = nn.Identity()
                    self.bn_upconv4 = nn.Identity()
                    self.conv_decode4 = nn.Conv2d(96, out_channel, 3, padding=1)
                else:
                    self.upconv4 = nn.ConvTranspose2d(96, 64, 2, 2)
                    self.bn_upconv4 = nn.BatchNorm2d(64)
                    self.conv_decode4 = nn.Conv2d(64, out_channel, 3, padding=1)

    def forward(self, representation):
        if self.num_classes is None:
            x = self.upconv1(representation)
            x = self.bn_upconv1(x)
            x = self.relu(x)
            x = self.conv_decode1(x)
            x = self.bn_decode1(x)
            x = self.relu(x)
            x = self.upconv2(x)
            x = self.bn_upconv2(x)
            x = self.relu(x)
            x = self.conv_decode2(x)

            x = self.bn_decode2(x)
            x = self.relu(x)
            x = self.upconv3(x)
            x = self.bn_upconv3(x)
            x = self.relu(x)
            x = self.conv_decode3(x)
            x = self.bn_decode3(x)
            x = self.relu(x)
            if not self.half_sized_output:
                x = self.upconv4(x)
                x = self.bn_upconv4(x)
                x = self.relu(x)
            x = self.conv_decode4(x)

        else:
            x = self.block12(representation)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


@register("celeb_a_decoder")
def build_decoder_for_celeb_a(in_channel, out_channel, drop_prob=0.5):
    return AttributeDecoder(in_channel, out_channel, drop_prob)


@register("celeb_a_decoder_no_drop")
def build_decoder_for_celeb_a(in_channel, out_channel):
    return AttributeDecoderNoDrop(in_channel, out_channel)


@register("taskonomy_decoder")
def build_decoder_for_taskonomy(out_channel, **kwargs):
    return ImgDecoder(out_channel=out_channel)
