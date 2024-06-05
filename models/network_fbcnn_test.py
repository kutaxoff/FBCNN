from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from timm.models.swin_transformer import SwinTransformer
from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.layers import to_2tuple, trunc_normal_


# class DepthwiseSeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#         super(DepthwiseSeparableConv2d, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
#         depthwise_separable_conv = nn.Sequential(self.depthwise, self.pointwise)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
    
# def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#     depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
#     pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
#     seq = nn.Sequential(depthwise, pointwise)
#     print()
#     print()
#     print('DSC:', sum(p.numel() for p in seq.parameters() if p.requires_grad))
#     print()
#     print()
#     return seq

class SwinBlock(nn.Module):
    def __init__(self, img_size=96, embed_dim=96, depths=[6,6,6,6], num_heads=[6,6,6,6]):
        super(SwinBlock, self).__init__()
        self.swin_transformer = SwinTransformer(
            img_size=img_size,  # Set this according to your input size
            patch_size=4,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

    def forward(self, x):
        print(x.shape, flush=True)
        B, C, H, W = x.shape
        # x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        # x = x.view(B, H * W, C)  # B, N, C where N = H * W

        # Apply Swin Transformer
        # x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape, flush=True)
        x = self.swin_transformer(x)
        print(x.shape, flush=True)
        x = x.permute(0, 3, 1, 2).contiguous()
        print(x.shape, flush=True)
        

        # Reshape back to match expected input for next Conv layers
        # x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        return x




'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            # conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            # print()
            # print()
            # print('Conv2d:', sum(p.numel() for p in conv.parameters() if p.requires_grad))
            # print()
            # print()
            # L.append(depthwise_separable_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
            # L.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))
        # elif t == 'I':
        #     L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'S':
            L.append(nn.SiLU(inplace=True))  # Using Swish activation
        # elif t == 'r':
        #     L.append(nn.ReLU(inplace=False))
        # elif t == 'L':
        #     L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        # elif t == 'l':
        #     L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        # elif t == '2':
        #     L.append(nn.PixelShuffle(upscale_factor=2))
        # elif t == '3':
        #     L.append(nn.PixelShuffle(upscale_factor=3))
        # elif t == '4':
        #     L.append(nn.PixelShuffle(upscale_factor=4))
        # elif t == 'U':
        #     L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        # elif t == 'u':
        #     L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        # elif t == 'v':
        #     L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        # elif t == 'M':
        #     L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        # elif t == 'A':
        #     L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
        
    # seq = sequential(*L)
    
    # print()
    # print()
    # print(' --- full --- :', sum(p.numel() for p in seq.parameters() if p.requires_grad))
    # print()
    # print()
    
    return sequential(*L)

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

class ResidualDenseBlockA(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(ResidualDenseBlockA, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()

        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=True)
            )
            self.conv_layers.append(nn.ReLU(inplace=True))

        self.conv_out = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        inputs = x
        for i, layer in enumerate(self.conv_layers):
            out = layer(inputs)
            if i % 2 == 0:  # Append only after convolution, not after activation
                inputs = torch.cat([inputs, out], dim=1)
        
        out = self.conv_out(inputs)
        return out + x  # Local residual learning

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32, num_layers=4, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock, self).__init__()

        self.num_layers = num_layers
        self.growth_channels = growth_channels

        self.dense_layers = nn.Sequential()
        for i in range(num_layers):
            seq_mode = mode #+ 'C' if i < num_layers - 1 else mode
            seq = conv(in_channels + i * growth_channels, growth_channels, kernel_size, stride, padding, bias, seq_mode, negative_slope)
            for layer in seq:
                self.dense_layers.append(layer)
            # self.dense_layers.append(seq)
        self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_channels, in_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        # inputs = [x]
        # for layer in self.dense_layers:
        #     out = layer(torch.cat(inputs, 1))
        #     inputs.append(out)
        # out = torch.cat(inputs, 1)
        # out = self.conv1x1(out)
        # return out + x
        inputs = x
        for i, layer in enumerate(self.dense_layers):
            out = layer(inputs)
            if i % 2 == 0:  # Append only after convolution, not after activation
                inputs = torch.cat([inputs, out], dim=1)
        
        out = self.conv1x1(inputs)
        return out + x  # Local residual learning
    
    
class ResidualSwinTransformerBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, num_layers=4):
        super(ResidualSwinTransformerBlock, self).__init__()
        self.swin_layer = sequential(*[SwinTransformerBlock(
            dim=in_channels,  # Adjust the dimension according to your channel size
            input_resolution=(96, 96),  # Adjust resolution based on your feature map size
            num_heads=8,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0
        ) for _ in range(num_layers)])

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 3, 1)  # Now shape is [batch_size, height, width, channels]
        x = self.swin_layer(x)
        x = x.permute(0, 3, 1, 2)
        res = self.conv(x)
        return input + res

# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)



class QFAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(QFAttention, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res(x) + beta
        return x + res
    
class CustomSwinTransformer(SwinTransformer):
    def __init__(self, *args, in_chans=512, **kwargs):
        super().__init__(*args, in_chans=in_chans, **kwargs)
        # Replace the patch embedding convolution
        self.patch_embed.proj = nn.Conv2d(in_chans, self.embed_dim, kernel_size=(1, 1), stride=(1, 1))


class FBCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, swin_layers=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(FBCNN, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.swin_layers = swin_layers
        self.nb = nb
        self.nc = nc
        
        # self.swin_block = sequential(*[SwinTransformerBlock(
        #     dim=nc[0],  # Adjust the dimension according to your channel size
        #     input_resolution=(96, 96),  # Adjust resolution based on your feature map size
        #     num_heads=8,
        #     window_size=7,
        #     shift_size=0,
        #     mlp_ratio=4.0
        # ) for _ in range(ns)])
        # # Initialize weights of the Swin Transformer Block
        # for param in self.swin_block.parameters():
        #     if param.ndim >= 2:  # Only apply to tensors with 2 or more dimensions
        #         trunc_normal_(param, std=.02)

        # self.swin_transformer = SwinTransformer(img_size=96,embed_dim=96, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6), window_size=7, num_classes=0, in_chans=64, patch_size=4)
        # self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResidualSwinTransformerBlock(nc[0], nc[0], num_layers=self.swin_layers) for _ in range(nb)],
            # *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            # *[ResidualDenseBlockA(nc[0]) for _ in range(nb)],
            # *[ResidualDenseBlock(nc[0], nc[0] // 2, bias=True, mode='C' + act_mode) for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        # self.swin1 = SwinBlock(embed_dim=nc[1])
        self.m_down2 = sequential(
            *[ResidualSwinTransformerBlock(nc[1], nc[1], num_layers=self.swin_layers) for _ in range(nb)],
            # *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            # *[ResidualDenseBlockA(nc[1]) for _ in range(nb)],
            # *[ResidualDenseBlock(nc[1], nc[1] // 2, bias=True, mode='C' + act_mode) for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        # self.swin2 = SwinBlock(embed_dim=nc[2])
        self.m_down3 = sequential(
            *[ResidualSwinTransformerBlock(nc[2], nc[2], num_layers=self.swin_layers) for _ in range(nb)],
            # *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            # *[ResidualDenseBlockA(nc[2]) for _ in range(nb)],
            # *[ResidualDenseBlock(nc[2], nc[2] // 2, bias=True, mode='C' + act_mode) for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode='2'))
        # print()
        # print(nc[3], flush=True)
        # print()
        # self.swin3 = SwinBlock(img_size=12, embed_dim=nc[3])
        # self.swin3 = SwinTransformer(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, num_classes=0, in_chans=512, patch_size=4)

        # self.swin_transformer = CustomSwinTransformer(
        #     embed_dim=nc[3],  # Match the number of input channels
        #     depths=[2, 2, 6, 2],  # Define the number of layers
        #     num_heads=[4, 8, 16, 32],  # Define the number of attention heads
        #     window_size=6,  # Smaller window size to match smaller input size
        #     # Adjust the patch size and other configurations as needed
        #     patch_size=1,  # Use patch size of 1 since input is already a feature map
        #     in_chans=nc[3],
        #     img_size=(12, 12)  # Define the new input size
        # )
        
        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])
            # *[ResidualDenseBlockA(nc[3]) for _ in range(nb)])
            # *[ResidualDenseBlock(nc[3], nc[3] // 2, bias=True, mode='C' + act_mode) for _ in range(nb)])

        self.m_body_decoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])
            # *[ResidualDenseBlockA(nc[3]) for _ in range(nb)])
            # *[ResidualDenseBlock(nc[3], nc[3] // 2, bias=True, mode='C' + act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = nn.ModuleList([upsample_block(nc[3], nc[2], bias=True, mode='2'),
                                  *[QFAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up2 = nn.ModuleList([upsample_block(nc[2], nc[1], bias=True, mode='2'),
                                  *[QFAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up1 = nn.ModuleList([upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                  *[QFAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])


        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')


        self.qf_pred = sequential(*[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
        # self.qf_pred = sequential(*[ResidualDenseBlockA(nc[3]) for _ in range(nb)],
        # self.qf_pred = sequential(*[ResidualDenseBlock(nc[3], nc[3] // 2, bias=True, mode='C' + act_mode) for _ in range(nb)],
                                  torch.nn.AdaptiveAvgPool2d((1,1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(512, 512), 
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 512),
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 1),
                                  nn.Sigmoid()
                                )

        self.qf_embed = sequential(torch.nn.Linear(1, 512),
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 512),
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 512),
                                  nn.ReLU()
                                )

        self.to_gamma_3 = sequential(torch.nn.Linear(512, nc[2]),nn.Sigmoid())
        self.to_beta_3 =  sequential(torch.nn.Linear(512, nc[2]),nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(512, nc[1]),nn.Sigmoid())
        self.to_beta_2 =  sequential(torch.nn.Linear(512, nc[1]),nn.Tanh())
        self.to_gamma_1 = sequential(torch.nn.Linear(512, nc[0]),nn.Sigmoid())
        self.to_beta_1 =  sequential(torch.nn.Linear(512, nc[0]),nn.Tanh())


    def forward(self, x, qf_input=None):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        # print(x.shape, flush=True)

        x1 = self.m_head(x)
        # print(x1.shape, flush=True)
        # batch_size, channels, height, width = x1.size()
        # x1 = x1.permute(0, 2, 3, 1)  # Now shape is [batch_size, height, width, channels]
        # x1 = self.swin_block(x1)
        # x1 = x1.permute(0, 3, 1, 2)
        # x1 = self.swin_transformer(x1)
        
        # print(x1.shape, flush=True)
        # x1 = x1.view(2, 64, 96, 96)
        
        # print(x1.shape, flush=True)
        # self.conv1x1(x1)
        
        x2 = self.m_down1(x1)
        # print(x2.shape, flush=True)
        # x2 = self.swin1(x2)
        x3 = self.m_down2(x2)
        # print(x3.shape, flush=True)
        # x3 = self.swin2(x3)
        x4 = self.m_down3(x3)
        # print(x4.shape, flush=True)
        # x4 = F.pad(x4, (0, 4, 0, 4), mode='constant', value=0)
        # x4_resized = F.interpolate(x4, size=(224, 224), mode='bilinear', align_corners=False)
        # print(x4.shape, flush=True)
        # x = self.swin3(x4_resized)
        # x = self.swin_transformer(x4)
        # x = x.view(x.size(0), 96, 4, 4)
        # print(x.shape, flush=True)
        # x = F.interpolate(x, size=(x4.size(2), x4.size(3)), mode='bilinear', align_corners=False)

        x = self.m_body_encoder(x4)
        # print(x.shape, flush=True)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)
        # print(x.shape, flush=True)
        qf_embedding = self.qf_embed(qf_input) if qf_input is not None else self.qf_embed(qf)
        gamma_3 = self.to_gamma_3(qf_embedding)
        beta_3 = self.to_beta_3(qf_embedding)

        gamma_2 = self.to_gamma_2(qf_embedding)
        beta_2 = self.to_beta_2(qf_embedding)

        gamma_1 = self.to_gamma_1(qf_embedding)
        beta_1 = self.to_beta_1(qf_embedding)


        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            x = self.m_up3[i+1](x, gamma_3,beta_3)

        x = x + x3

        x = self.m_up2[0](x)
        for i in range(self.nb):
            x = self.m_up2[i+1](x, gamma_2, beta_2)
        x = x + x2

        x = self.m_up1[0](x)
        for i in range(self.nb):
            x = self.m_up1[i+1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x, qf

if __name__ == "__main__":
    x = torch.randn(1, 3, 96, 96)#.cuda()#.to(torch.device('cuda'))
    fbar=FBAR()
    y,qf = fbar(x)
    print(y.shape,qf.shape)
