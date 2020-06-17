import torch
from torch import nn

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some convolutional blocks
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    '''
    upblock that does one up-sampling step: Upsample -> COnvBlock -> ConvBlock
    '''
    DEPTH = 6
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None, upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ResNetUNet(nn.Module):
    DEPTH=6
    def __init__(self, encoder, n_class=2):
        super().__init__()

        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(encoder.children()))[:3]
        self.input_pool  = list(encoder.children())[3]

        for bottleneck in list(encoder.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)

        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(in_channels=128 + 64,
                                 out_channels=128,
                                 up_conv_in_channels=256,
                                 up_conv_out_channels=128))
        up_blocks.append(UpBlock(in_channels=64 + 4,
                                 out_channels=64,
                                 up_conv_in_channels=128,
                                 up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_class, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):

        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)

        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (ResNetUNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{ResNetUNet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        x = self.out(x)
        del pre_pools

        return x