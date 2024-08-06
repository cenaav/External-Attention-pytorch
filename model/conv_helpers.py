from .conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from .conv.MBConv import MBConvBlock
from .conv.Involution import Involution
from .conv.DynamicConv import *
from .conv.CondConv import *


def create_conv_block(class_name, in_channels=16, out_channels=16, channel_wh=512):  
    match class_name:
        case 'DepthwiseSeparableConvolution':
            return DepthwiseSeparableConvolution(in_channels, out_channels)
        case 'MBConvBlock':
            return MBConvBlock(ksize=3, input_filters=in_channels, output_filters=out_channels, image_size=channel_wh)
        case 'Involution':
            return Involution(kernel_size=3, in_channel=in_channels, stride=1)
        case 'DynamicConv':
            return DynamicConv(in_planes=in_channels, out_planes=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        case 'CondConv':
            return CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        case _:
            raise ValueError(f"No class found for the name {class_name}")
