from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np


def sequential(*args):
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
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
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


class FBCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R'):
        super(FBCNN, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        
        downsample_block = downsample_strideconv

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode='2'))

        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_body_decoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        upsample_block = upsample_convtranspose

        self.m_up3 = nn.ModuleList([upsample_block(nc[3], nc[2], bias=True, mode='2'),
                                  *[QFAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up2 = nn.ModuleList([upsample_block(nc[2], nc[1], bias=True, mode='2'),
                                  *[QFAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up1 = nn.ModuleList([upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                  *[QFAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])


        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')


        self.qf_pred = sequential(*[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
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

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)
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
