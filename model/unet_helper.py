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
from .attention.ParNetAttention import ParNetAttention
from .attention.UFOAttention import UFOAttention
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
from .conv.DynamicConv import DynamicConv
from .conv.CondConv import CondConv



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
        l2 = d_k * d_k
        l3 = d_v * d_v
        self.model = EMSA(d_model=l, d_k=l2, d_v=l3, h=h, H=H, W=W, ratio=ratio, apply_transform=apply_transform)
        
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        #input_reshaped = input.view(batch_size, channels, height * width).permute(0, 1, 2)
        input_reshaped = input.view(batch_size, channels, -1)
        # Apply attention
        output = self.model(input_reshaped, input_reshaped, input_reshaped)
        output = output.view(batch_size, channels, height, width)
        return output
    

class MUSEAttentionHelper(nn.Module):

    def __init__(self, d_model=512, d_k=512, d_v=512, h=8, dropout=0.1):
        super().__init__()
        l = d_model * d_model
        l2 = d_k * d_k
        l3 = d_v * d_v
        self.model = MUSEAttention(d_model=l, d_k=l2, d_v=l3, h=h, dropout=dropout)
        
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
        self.model = AFT_FULL(l, n, simple)
        
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
    


def create_attention_block(class_name, ch_in=16, ch_out=16, ch_wh=512):  
    
    match class_name:
        #------------------------------------------------------------------------------
        #Attention
        case 'ExternalAttention':
            return ExternalAttention(d_model=ch_wh, S=8)
        case 'ScaledDotProductAttention':
            return ScaledDotProductAttentionHelper(d_model=ch_wh, d_k=ch_wh, d_v=ch_wh, h=8, dropout=0.1)
        case 'SimplifiedScaledDotProductAttention':
            return SimplifiedScaledDotProductAttentionHelper(d_model=ch_in, h=8, dropout=0.1)
        case 'SEAttention':    
            return SEAttention(channel=ch_in, reduction=16)
        case 'SKAttention':
            return SKAttention(channel=ch_in, kernels=[1,3,5,7], reduction=16, group=1, L=32)
        case 'CBAMBlock':
            r = 8 if ch_in > 8 else ch_in
            return CBAMBlock(channel=ch_in, reduction=r, kernel_size=3)
        case 'BAMBlock':
            # throw an error if batch number is equal to 1
            return BAMBlock(channel=ch_in, reduction=8, dia_val=1)
        case 'ECAAttention':
            return ECAAttention(kernel_size=3)
        case 'DAModule':
            return DAModule(d_model=ch_in, kernel_size=3, H=ch_wh, W=ch_wh)
        case 'PSA':
            r = 8 if ch_in > 8 else ch_in
            return PSA(channel=ch_in, reduction=r, S=1)
        case 'EMSA':
            return EMSAHelper(d_model=ch_wh, d_k=ch_wh, d_v=ch_wh, h=8, H=8, W=8, ratio=1, apply_transform=True)
        case 'ShuffleAttention':
            if (ch_in == 1) or (ch_in == 3) or (ch_in % 2 !=0):
                print("Wrong Input Parameter")
                return None
            g = int(ch_in/2)
            return ShuffleAttention(channel=ch_in, reduction=16, G=g)
        case 'MUSEAttention':
            return MUSEAttentionHelper(d_model=ch_wh, d_k=ch_wh, d_v=ch_wh, h=8, dropout=0.1)
        case 'SpatialGroupEnhance':
            if ch_in % 2 !=0:
                g = ch_in
            else:
                g = int(ch_in/2)
            return SpatialGroupEnhance(groups=g)
        case 'DoubleAttention':
            return DoubleAttention(in_channels=ch_in, c_m=32, c_n=32, reconstruct=True)
        case 'AFT_FULL':
            return AFTFULLHelper(d_model=ch_wh, n=ch_in, simple=False)
        case 'OutlookAttention':
            return OutlookAttention(dim=ch_wh, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False, attn_drop=0.1)
        case 'WeightedPermuteMLP':
            return WeightedPermuteMLP(dim=ch_in, seg_dim=8, qkv_bias=False, proj_drop=0.)
        case 'CoAtNet':
            raise ValueError(f"No class found for the name {class_name}")
        case 'HaloAttention':
            return HaloAttention(dim=ch_in, block_size=2, halo_size=1, dim_head=64, heads=8)
        case 'ParallelPolarizedSelfAttention':
            return ParallelPolarizedSelfAttention(channel=ch_in)
        case 'SequentialPolarizedSelfAttention':
            return SequentialPolarizedSelfAttention(channel=ch_in)
        case 'CoTAttention':
            return CoTAttention(dim=ch_in, kernel_size=3)
        case 'ResidualAttention':
            raise ValueError(f"No class found for the name {class_name}")
        case 'S2Attention':
            return S2Attention(channels=ch_in)
        case 'GFNet':
            raise ValueError(f"No class found for the name {class_name}")
        case 'TripletAttention':
            return TripletAttention(no_spatial=False)
        case 'CoordAtt':
            return CoordAtt(inp=ch_in, oup=ch_out, reduction=32)
        case 'MobileViTAttention':
            return MobileViTAttention(in_channel=ch_in, dim=ch_wh, kernel_size=3, patch_size=2)
        case 'ParNetAttention':
            return ParNetAttention(channel=ch_in)
        case 'UFOAttention':
            return UFOAttentionHelper(d_model=ch_wh, d_k=32, d_v=32, h=8, dropout=0.1)
        case 'ACmix':
            return ACmix(in_planes=ch_in, out_planes=ch_out, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1)
        case 'MobileViTv2Attention':
            return MobileViTv2AttentionHelper(d_model=ch_wh)
        case 'DAT':
            raise ValueError(f"No class found for the name {class_name}")
        case 'CrossFormer':
            raise ValueError(f"No class found for the name {class_name}")
        case 'MOATransformer':
            raise ValueError(f"No class found for the name {class_name}")
        case 'CrissCrossAttention':
            return CrissCrossAttention(in_dim=ch_in)
        case 'AxialImageTransformer':
            return AxialImageTransformer(dim=ch_in, depth=12, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None)
        #------------------------------------------------------------------------------
        #MLP
        case 'sMLPBlock':
            return sMLPBlock(h=ch_wh, w=ch_wh, c=ch_in)
        #------------------------------------------------------------------------------
        #Conv
        case 'DepthwiseSeparableConvolution':
            return DepthwiseSeparableConvolution(in_ch=ch_in, out_ch=ch_out, kernel_size=3, stride=1, padding=1)
        case 'MBConvBlock':
            return MBConvBlock(ksize=3, input_filters=ch_in, output_filters=ch_out, expand_ratio=1, stride=1, image_size=ch_wh)
        case 'Involution':
            return Involution(kernel_size=3, in_channel=ch_in, stride=1, group=1, ratio=4)
        case 'DynamicConv':
            return DynamicConv(in_planes=ch_in, out_planes=ch_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, K=4, temprature=30, ratio=4, init_weight=True)
        case 'CondConv':
            return CondConv(in_planes=ch_in, out_planes=ch_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, K=4, init_weight=True)
        case _:
            raise ValueError(f"No class found for the name {class_name}")
