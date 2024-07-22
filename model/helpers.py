from attention.ExternalAttention import ExternalAttention
from attention.SelfAttention import ScaledDotProductAttention
from attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from attention.SEAttention import SEAttention
from attention.SKAttention import SKAttention
from attention.CBAM import CBAMBlock
from attention.BAM import BAMBlock
from attention.ECAAttention import ECAAttention
from attention.DANet import DAModule
from attention.PSA import PSA
from attention.EMSA import EMSA
from attention.ShuffleAttention import ShuffleAttention
from attention.MUSEAttention import MUSEAttention
from attention.SGE import SpatialGroupEnhance
from attention.A2Atttention import DoubleAttention
from attention.AFT import AFT_FULL
from attention.OutlookAttention import OutlookAttention
from attention.ViP import WeightedPermuteMLP
from attention.CoAtNet import CoAtNet
from attention.HaloAttention import HaloAttention
from attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
from attention.CoTAttention import CoTAttention
from attention.ResidualAttention import ResidualAttention
from attention.S2Attention import S2Attention
from attention.gfnet import GFNet
from attention.TripletAttention import TripletAttention
from attention.CoordAttention import CoordAtt
from attention.MobileViTAttention import MobileViTAttention
from attention.ParNetAttention import *
from attention.UFOAttention import *
from attention.ACmixAttention import ACmix
from attention.MobileViTv2Attention import MobileViTv2Attention
from attention.DAT import DAT
from attention.Crossformer import CrossFormer
from attention.MOATransformer import MOATransformer
from attention.CrissCrossAttention import CrissCrossAttention
from attention.Axial_attention import AxialImageTransformer

from backbone.resnet import ResNet50,ResNet101,ResNet152
from backbone.resnext import ResNeXt50,ResNeXt101,ResNeXt152
from backbone.MobileViT import *
from backbone.ConvMixer import *
from backbone.ShuffleTransformer import ShuffleTransformer
from backbone.ConTNet import ConTNet
from backbone.HATNet import HATNet
from backbone.CoaT import CoaT
from backbone.PVT import PyramidVisionTransformer
from backbone.CPVT import CPVTV2
from backbone.PIT import PoolingTransformer
from backbone.CrossViT import VisionTransformer
from backbone.TnT import TNT
from backbone.DViT import DeepVisionTransformer
from backbone.CeiT import CeIT
from backbone.ConViT import VisionTransformer
from backbone.CaiT import CaiT
from backbone.PatchConvnet import PatchConvnet
from backbone.DeiT import DistilledVisionTransformer
from backbone.PatchConvnet import PatchConvnet
from backbone.VOLO import VOLO
from backbone.Container import VisionTransformer
from backbone.CMT import CMT_Tiny
from backbone.LeViT import *
from backbone.EfficientFormer import EfficientFormer
from backbone.convnextv2 import convnextv2_atto

from mlp.repmlp import RepMLP
from mlp.mlp_mixer import MlpMixer
from mlp.resmlp import ResMLP
from mlp.g_mlp import gMLP
from mlp.sMLP_block import sMLPBlock
from mlp.vip_mlp import VisionPermutator

from rep.repvgg import RepBlock
from rep.acnet import ACNet
from rep.ddb import transI_conv_bn
from rep.ddb import transII_conv_branch
from rep.ddb import transIII_conv_sequential
from rep.ddb import transIV_conv_concat
from rep.ddb import transV_avg
from rep.ddb import transVI_conv_scale

from attention.MobileViTv2Attention import *
from conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from conv.MBConv import MBConvBlock
from conv.Involution import Involution
from conv.DynamicConv import *
from conv.CondConv import *


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
