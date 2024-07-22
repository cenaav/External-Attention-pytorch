from model.attention.ExternalAttention import ExternalAttention
from model.attention.SelfAttention import ScaledDotProductAttention
from model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from model.attention.SEAttention import SEAttention
from model.attention.SKAttention import SKAttention
from model.attention.CBAM import CBAMBlock
from model.attention.BAM import BAMBlock
from model.attention.ECAAttention import ECAAttention
from model.attention.DANet import DAModule
from model.attention.PSA import PSA
from model.attention.EMSA import EMSA
from model.attention.ShuffleAttention import ShuffleAttention
from model.attention.MUSEAttention import MUSEAttention
from model.attention.SGE import SpatialGroupEnhance
from model.attention.A2Atttention import DoubleAttention
from model.attention.AFT import AFT_FULL
from model.attention.OutlookAttention import OutlookAttention
from model.attention.ViP import WeightedPermuteMLP
from model.attention.CoAtNet import CoAtNet
from model.attention.HaloAttention import HaloAttention
from model.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
from model.attention.CoTAttention import CoTAttention
from model.attention.ResidualAttention import ResidualAttention
from model.attention.S2Attention import S2Attention
from model.attention.gfnet import GFNet
from model.attention.TripletAttention import TripletAttention
from model.attention.CoordAttention import CoordAtt
from model.attention.MobileViTAttention import MobileViTAttention
from model.attention.ParNetAttention import *
from model.attention.UFOAttention import *
from model.attention.ACmixAttention import ACmix # fix this import in command
from model.attention.MobileViTv2Attention import MobileViTv2Attention
from model.attention.DAT import DAT
from model.attention.Crossformer import CrossFormer
from model.attention.MOATransformer import MOATransformer
from model.attention.CrissCrossAttention import CrissCrossAttention
from model.attention.Axial_attention import AxialImageTransformer

from model.backbone.resnet import ResNet50,ResNet101,ResNet152
from model.backbone.resnext import ResNeXt50,ResNeXt101,ResNeXt152
from model.backbone.MobileViT import *
from model.backbone.ConvMixer import *
from model.backbone.ShuffleTransformer import ShuffleTransformer
from model.backbone.ConTNet import ConTNet
from model.backbone.HATNet import HATNet
from model.backbone.CoaT import CoaT
from model.backbone.PVT import PyramidVisionTransformer
from model.backbone.CPVT import CPVTV2
from model.backbone.PIT import PoolingTransformer
from model.backbone.CrossViT import VisionTransformer
from model.backbone.TnT import TNT
from model.backbone.DViT import DeepVisionTransformer
from model.backbone.CeiT import CeIT
from model.backbone.ConViT import VisionTransformer
from model.backbone.CaiT import CaiT
from model.backbone.PatchConvnet import PatchConvnet
from model.backbone.DeiT import DistilledVisionTransformer
from model.backbone.PatchConvnet import PatchConvnet
from model.backbone.VOLO import VOLO
from model.backbone.Container import VisionTransformer
from model.backbone.CMT import CMT_Tiny
from model.backbone.LeViT import *
from model.backbone.EfficientFormer import EfficientFormer
from model.backbone.convnextv2 import convnextv2_atto

from model.mlp.repmlp import RepMLP
from model.mlp.mlp_mixer import MlpMixer
from model.mlp.resmlp import ResMLP
from model.mlp.g_mlp import gMLP
from model.mlp.sMLP_block import sMLPBlock
from model.mlp.vip_mlp import VisionPermutator

from model.rep.repvgg import RepBlock
from model.rep.acnet import ACNet
from model.rep.ddb import transI_conv_bn
from model.rep.ddb import transII_conv_branch
from model.rep.ddb import transIII_conv_sequential
from model.rep.ddb import transIV_conv_concat
from model.rep.ddb import transV_avg
from model.rep.ddb import transVI_conv_scale

from model.attention.MobileViTv2Attention import *
from model.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from model.conv.MBConv import MBConvBlock
from model.conv.Involution import Involution
from model.conv.DynamicConv import *
from model.conv.CondConv import *


def create_attention_block(class_name, in_channels=16, out_channels=16, channel_wh=512):  
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
