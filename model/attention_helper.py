from .attention.ExternalAttention import ExternalAttention
from .attention.SelfAttention import ScaledDotProductAttention
from .attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from .attention.SEAttention import SEAttention
from .attention.SKAttention import SKAttention
from .attention.CBAM import CBAMBlock
from .attention.BAM import BAMBlock
from .attention.ECAAttention import ECAAttention
from .attention.DANet import DAModule
from .attention.PSA import PSA
from .attention.EMSA import EMSA
from .attention.ShuffleAttention import ShuffleAttention
from .attention.MUSEAttention import MUSEAttention
from .attention.SGE import SpatialGroupEnhance
from .attention.A2Atttention import DoubleAttention
from .attention.AFT import AFT_FULL
from .attention.OutlookAttention import OutlookAttention
from .attention.ViP import WeightedPermuteMLP
from .attention.CoAtNet import CoAtNet
from .attention.HaloAttention import HaloAttention
from .attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention, SequentialPolarizedSelfAttention
from .attention.CoTAttention import CoTAttention
from .attention.ResidualAttention import ResidualAttention
from .attention.S2Attention import S2Attention
from .attention.gfnet import GFNet
from .attention.TripletAttention import TripletAttention
from .attention.CoordAttention import CoordAtt
from .attention.MobileViTAttention import MobileViTAttention
from .attention.ParNetAttention import *
from .attention.UFOAttention import *
from .attention.ACmixAttention import ACmix # fix this import in command
from .attention.MobileViTv2Attention import MobileViTv2Attention
from .attention.DAT import DAT
from .attention.Crossformer import CrossFormer
from .attention.MOATransformer import MOATransformer
from .attention.CrissCrossAttention import CrissCrossAttention
from .attention.Axial_attention import AxialImageTransformer
##
from .backbone.resnet import ResNet50, ResNet101, ResNet152
from .backbone.resnext import ResNeXt50, ResNeXt101, ResNeXt152
from .backbone.MobileViT import *
from .backbone.ConvMixer import *
from .backbone.ShuffleTransformer import ShuffleTransformer
from .backbone.ConTNet import ConTNet
from .backbone.HATNet import HATNet
from .backbone.CoaT import CoaT
from .backbone.PVT import PyramidVisionTransformer
from .backbone.CPVT import CPVTV2
from .backbone.PIT import PoolingTransformer
from .backbone.CrossViT import VisionTransformer
from .backbone.TnT import TNT
from .backbone.DViT import DeepVisionTransformer
from .backbone.CeiT import CeIT
from .backbone.ConViT import VisionTransformer
from .backbone.CaiT import CaiT
from .backbone.PatchConvnet import PatchConvnet
from .backbone.DeiT import DistilledVisionTransformer
from .backbone.PatchConvnet import PatchConvnet
from .backbone.VOLO import VOLO
from .backbone.Container import VisionTransformer
from .backbone.CMT import CMT_Tiny
from .backbone.LeViT import *
from .backbone.EfficientFormer import EfficientFormer
from .backbone.convnextv2 import convnextv2_atto
##
from .mlp.repmlp import RepMLP
from .mlp.mlp_mixer import MlpMixer
from .mlp.resmlp import ResMLP
from .mlp.g_mlp import gMLP
from .mlp.sMLP_block import sMLPBlock
from .mlp.vip_mlp import VisionPermutator
##
from .rep.repvgg import RepBlock
from .rep.acnet import ACNet
from .rep.ddb import transI_conv_bn
from .rep.ddb import transII_conv_branch
from .rep.ddb import transIII_conv_sequential
from .rep.ddb import transIV_conv_concat
from .rep.ddb import transV_avg
from .rep.ddb import transVI_conv_scale
##
from .conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from .conv.MBConv import MBConvBlock
from .conv.Involution import Involution
from .conv.DynamicConv import *
from .conv.CondConv import *



class ScaledDotProductAttentionHelper(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        super().__init__()
        l = d_model * d_model
        self.model = ScaledDotProductAttention(d_model=l, d_k=l, d_v=l, h=h, dropout=dropout)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        #input_reshaped = input.view(batch_size, channels, height * width).permute(0, 1, 2)
        input_reshaped = input.view(batch_size, channels, -1)
        # Apply attention
        output = self.model(input_reshaped, input_reshaped, input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output


class SimplifiedScaledDotProductAttentionHelper(nn.Module):

    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()

        self.model = SimplifiedScaledDotProductAttention(d_model, h, dropout)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # Reshape input tensor to (batch_size, height * width, channels)
        input_reshaped = input.view(batch_size, channels, height * width).permute(0, 2, 1)
        #print(input_reshaped.shape)
        output = self.model(input_reshaped, input_reshaped, input_reshaped)
        # Reshape output tensor back to (batch, channel, height, width)
        output = output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return output
    

class EMSAHelper(nn.Module):

    def __init__(self, d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=1, apply_transform=True):
        super().__init__()
        l = d_model * d_model
        self.model = EMSA(d_model=d_model, d_k=d_k, d_v=d_v, 
                          h=h, H=H, W=W, ratio=ratio, apply_transform=apply_transform)
        
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        #input_reshaped = input.view(batch_size, channels, height * width).permute(0, 1, 2)
        input_reshaped = input.view(batch_size, channels, -1)
        print(input_reshaped.shape)
        # Apply attention
        output = self.model(input_reshaped, input_reshaped, input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output
    

class MUSEAttentionHelper(nn.Module):

    def __init__(self, d_model=512, d_k=512, d_v=512, h=8, dropout=0.1):
        super().__init__()
        l = d_model * d_model
        self.model = MUSEAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dropout=dropout)
        
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        #input_reshaped = input.view(batch_size, channels, height * width).permute(0, 1, 2)
        input_reshaped = input.view(batch_size, channels, -1)
        # Apply attention
        output = self.model(input_reshaped, input_reshaped, input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output
    

class AFTFULLHelper(nn.Module):

    def __init__(self, d_model=128, n=16, simple=False):
        super().__init__()
        l = d_model * d_model
        self.model = AFT_FULL(d_model, n, simple)
        
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        #input_reshaped = input.view(batch_size, channels, height * width).permute(0, 1, 2)
        input_reshaped = input.view(batch_size, channels, -1)
        # Apply attention
        output = self.model(input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output
    
    
class UFOAttentionHelper(nn.Module):

    def __init__(self, d_model=32, d_k=32, d_v=32, h=8, dropout=0.1):
        super().__init__()
        l = d_model * d_model
        self.model = UFOAttention(l, d_k, d_v, h, dropout)
        
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        input_reshaped = input.view(batch_size, channels, -1)
        # Apply attention
        output = self.model(input_reshaped, input_reshaped, input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output
    

class MobileViTv2AttentionHelper(nn.Module):

    def __init__(self, d_model=32):
        super().__init__()
        l = d_model * d_model
        self.model = MobileViTv2Attention(l)
        
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        input_reshaped = input.view(batch_size, channels, -1)
        # Apply attention
        output = self.model(input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output
    


def create_attention_block(class_name, channel=16, 
                           d_model=32, d_k=32, d_v=32, h=8, S=8, dropout=0.1, reduction=16,
                           kernels=[1,3,5,7], group=1, L=32, kernel_size=49, dia_val=2,
                           H=32, W=32, ratio=1, apply_transform=True, G=8,
                           in_channels=16, out_channels=16, c_m=32, c_n=32, reconstruct=True, n=3, simple=False,
                           dim=32, num_heads=1, padding=1, stride=1, qkv_bias=False, attn_drop=0.1,
                           seg_dim=8, proj_drop=0., block_size=3, halo_size=1, dim_head=64, heads=8,
                           no_spatial=False, patch_size=16, kernel_att=7, kernel_conv=3, dilation=1,
                           depth=12, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None,
                           bias=False, image_size=224, K=4, temprature=30, init_weight=True): 
    
    match class_name:
        #------------------------------------------------------------------------------
        #Attention
        case 'ExternalAttention':
            return ExternalAttention(d_model, S)
        case 'ScaledDotProductAttention':
            return ScaledDotProductAttentionHelper(d_model, d_k, d_v, h, dropout)
        case 'SimplifiedScaledDotProductAttention':
            return SimplifiedScaledDotProductAttentionHelper(d_model, h, dropout)
        case 'SEAttention':    
            return SEAttention(channel, reduction)
        case 'SKAttention':
            return SKAttention(channel, kernels, reduction, group, L)
        case 'CBAMBlock':
            return CBAMBlock(channel, reduction, kernel_size)
        case 'BAMBlock':
            return BAMBlock(channel, reduction, dia_val)
        case 'ECAAttention':
            return ECAAttention(kernel_size)
        case 'DAModule':
            return DAModule(d_model, kernel_size, H, W)
        case 'PSA':
            return PSA(channel, reduction, S)
        case 'EMSA':
            return EMSAHelper(d_model, d_k, d_v, h, H, W, ratio, apply_transform)
        case 'ShuffleAttention':
            return ShuffleAttention(channel, reduction, G)
        case 'MUSEAttention':
            return MUSEAttentionHelper(d_model, d_k, d_v, h, dropout)
        case 'SpatialGroupEnhance':
            return SpatialGroupEnhance(group)
        case 'DoubleAttention':
            return DoubleAttention(in_channels, c_m, c_n, reconstruct)
        case 'AFT_FULL':
            return AFTFULLHelper(d_model, n, simple)
        case 'OutlookAttention':
            return OutlookAttention(dim, num_heads, kernel_size, padding, stride, qkv_bias, attn_drop)
        case 'WeightedPermuteMLP':
            return WeightedPermuteMLP(dim, seg_dim, qkv_bias, proj_drop)
        case 'CoAtNet':
            raise ValueError(f"No class found for the name {class_name}")
        case 'HaloAttention':
            return HaloAttention(dim, block_size, halo_size, dim_head, heads)
        case 'ParallelPolarizedSelfAttention':
            return ParallelPolarizedSelfAttention(channel)
        case 'SequentialPolarizedSelfAttention':
            return SequentialPolarizedSelfAttention(channel)
        case 'CoTAttention':
            return CoTAttention(dim, kernel_size)
        case 'ResidualAttention':
            raise ValueError(f"No class found for the name {class_name}")
        case 'S2Attention':
            return S2Attention(channel)
        case 'GFNet':
            raise ValueError(f"No class found for the name {class_name}")
        case 'TripletAttention':
            return TripletAttention(no_spatial)
        case 'CoordAtt':
            return CoordAtt(in_channels, out_channels, reduction)
        case 'MobileViTAttention':
            return MobileViTAttention(in_channels, dim, kernel_size, patch_size)
        case 'ParNetAttention':
            return ParNetAttention(channel)
        case 'UFOAttention':
            return UFOAttentionHelper(d_model, d_k, d_v, h, dropout)
        case 'ACmix':
            return ACmix(in_channels, out_channels, kernel_att, heads, kernel_conv, stride, dilation)
        case 'MobileViTv2Attention':
            return MobileViTv2AttentionHelper(d_model)
        case 'DAT':
            raise ValueError(f"No class found for the name {class_name}")
        case 'CrossFormer':
            raise ValueError(f"No class found for the name {class_name}")
        case 'MOATransformer':
            raise ValueError(f"No class found for the name {class_name}")
        case 'CrissCrossAttention':
            return CrissCrossAttention(channel)
        case 'AxialImageTransformer':
            return AxialImageTransformer(dim, depth, heads, dim_heads, dim_index, reversible, axial_pos_emb_shape)
        #------------------------------------------------------------------------------
        #MLP
        case 'sMLPBlock':
            return sMLPBlock(h=H, w=W, c=channel)
        #------------------------------------------------------------------------------
        #Conv
        case 'DepthwiseSeparableConvolution':
            return DepthwiseSeparableConvolution(in_channels, out_channels, kernel_size, stride, padding)
        case 'MBConvBlock':
            return MBConvBlock(kernel_size, in_channels, out_channels, ratio, stride, image_size)
        case 'Involution':
            return Involution(kernel_size, in_channels, stride, group, ratio)
        case 'DynamicConv':
            return DynamicConv(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias, K, temprature, ratio, init_weight)
        case 'CondConv':
            return CondConv(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias, K, init_weight)
        case _:
            raise ValueError(f"No class found for the name {class_name}")
