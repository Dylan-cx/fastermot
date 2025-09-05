from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
# from models.networks.attentions import get_attention_module
import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from mmcv.ops import DeformConv2dPack as DCN
#from DCN.dcn_v2 import DCN

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
import copy
from torch import Tensor
try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# get_model_url 下载预训练权重
def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))

# P卷积，通道不变，dim
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        # 进行部分卷积的维度
        # //是取整除法，n_div是进行部分卷积的比例（四分之一？）
        # dim是in_channel
        self.dim_conv3 = dim // n_div
        # 剩余不变的维度
        self.dim_untouched = dim - self.dim_conv3
        # nn.Conv2d参数：in_channel，out_channel，kernel_size,stride,padding
        # 输入输出通道数不变
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        self.forward_type = forward

        # if forward == 'slicing':
        #     self.forward = self.forward_slicing
        # elif forward == 'split_cat':
        #     self.forward = self.forward_split_cat
        # else:
        #     raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_type == 'slicing':
            # only for inference
            x = x.clone()  # !!! Keep the original input intact for the residual connection later
            x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        elif self.forward_type == 'split_cat':
            # for training/inference
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            x = torch.cat((x1, x2), 1)

        return x


# Faster Block
# 通道不变，dim
class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # torch.nn.Identity( ) 相当于一个恒等函数 f(x) = x 这个函数相当于输入什么就输出什么
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        # mlp层中间的尺度膨胀（s网络是2倍）率为mlp_ratio，膨胀通道为mlp_hidden_dim
        mlp_hidden_dim = int(dim * mlp_ratio)

        # norm_layer正则化层，一般是bn
        # act_layer是激活函数层，一般是relu
        # mlp_layer经过conv-bn-relu-conv，整体输入输出通道不变（dim）
        # mlp就是fasternet的两层1x1点卷积
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        # layer_scale_init_value初始化可学习的权重
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward_type = 'forward_layer_scale'
        else:
            self.forward_type = 'forward'

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_type == 'forward_layer_scale':
            shortcut = x
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(
                self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))

        elif self.forward_type == 'forward':
            shortcut = x
            # P卷积
            x = self.spatial_mixing(x)
            # 残差（原本的x+P卷积和两层点卷积后的x）
            x = shortcut + self.drop_path(self.mlp(x))
        return x


# stage，由depth决定深度
# 通道不变，dim
class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        # 每个stage都会堆叠多个FasterNet Block，个数由depth决定
        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


# conv-bn，通道由in_chans变为embed_dim
class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        # in_chans为输入通道，embed_dim为输出通道
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=1, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


# conv-bn，通道由dim变为2倍dim
class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        # 通道变为2倍
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# init用于初始化（用什么卷积，用什么relu），forword定义网络如何传播，即网络的结构
# BasicBlock：一个基本的残差单元，包括conv→bn→relu→conv→bn→残差连接→relu
# 师兄的注意力加在这里，将第二次conv和bn替换成带注意力的conv和bn
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, inplanes, planes, stride=1, dilation=1, attention_module=None):
#         super(BasicBlock, self).__init__()
#         # inplane是in_channel,plane是out_channel
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.stride = stride
#
#         if attention_module is not None:
#             if type(attention_module) == functools.partial:
#                 module_name = attention_module.func.get_module_name()
#             else:
#                 module_name = attention_module.get_module_name()
#
#
#             if module_name == "simam":
#                 self.conv2 = nn.Sequential(
#                     self.conv2,
#                     attention_module(planes * self.expansion)
#                 )
#             else:
#                 self.bn2 = nn.Sequential(
#                     self.bn2,
#                     attention_module(planes * self.expansion)
#                 )
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# # 比basicblock多了一次cbl：conv→bn→relu→conv→bn→relu→conv→bn→残差连接→relu
# # 瓶颈，输出通道减少？
# class Bottleneck(nn.Module):
#     expansion = 2
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(Bottleneck, self).__init__()
#         expansion = Bottleneck.expansion
#         bottle_planes = planes // expansion
#         self.conv1 = nn.Conv2d(inplanes, bottle_planes,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes, planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# # conv→bn→relu→conv→bn→relu→conv→bn→残差连接→relu
# # bottle_planes随planes改变
# class BottleneckX(nn.Module):
#     expansion = 2
#     cardinality = 32
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BottleneckX, self).__init__()
#         cardinality = BottleneckX.cardinality
#         # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
#         # bottle_planes = dim * cardinality
#         bottle_planes = planes * cardinality // 32
#         self.conv1 = nn.Conv2d(inplanes, bottle_planes,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
#                                stride=stride, padding=dilation, bias=False,
#                                dilation=dilation, groups=cardinality)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes, planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# # Root聚合不同尺度的特征信息
# # Root：把输入张量列表按通道拼接，然后执行Conv和Relu
# # 改变了通道数
# class Root(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, residual):
#         super(Root, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, 1,
#             stride=1, bias=False, padding=(kernel_size - 1) // 2)
#         self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.residual = residual
#
#     # *x代表多个输入
#     def forward(self, *x):
#         children = x
#         # 从channel维度上把多层次特征信息concate
#         x = self.conv(torch.cat(x, 1))
#         x = self.bn(x)
#         # 类似于resnet的残差
#         if self.residual:
#             x += children[0]
#         x = self.relu(x)
#
#         return x
#
#
# class Tree(nn.Module):
#     #levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root
#     #level2 1 BasicBlock 32 64 2 false false
#     #level3 2 BasicBlock 64 128 2 True false
#     def __init__(self, levels, block, in_channels, out_channels, stride=1,
#                  level_root=False, root_dim=0, root_kernel_size=1,
#                  dilation=1, root_residual=False):
#         super(Tree, self).__init__()
#         # root_dim是root的输入通道数
#         # level2 concat两个block输出，因此输入维度变为2个
#         if root_dim == 0:
#             root_dim = 2 * out_channels
#         if level_root:
#             '''
#                     如果是树根，除了要concate自己的两个子树之外，还要链接上一个s树的downsample作用下的输出，
#                     如forward中children = [bottom,x1,x2]，因此加in_channels
#                     '''
#             # level3 的root_dim需要加上level2的root输出维度
#             root_dim += in_channels
#         # levels=1，即第1层root，它子节点是两个block，block=basicblock
#         if levels == 1:
#             self.tree1 = block(in_channels, out_channels, stride,
#                                dilation=dilation)
#             self.tree2 = block(out_channels, out_channels, 1,
#                                dilation=dilation)
#         # 递归调用Tree类
#         else:
#             # root_dim = 0进入迭代后为2 * out_channels
#             # stride = 2
#             self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
#                               stride, root_dim=0,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation, root_residual=root_residual)
#             # 这里的root_dim加上了tree1的输出通道数
#             # stride变成1了，不再downsample
#             # 输入输出维度相同，不再project映射
#             self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
#                               root_dim=root_dim + out_channels,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation, root_residual=root_residual)
#         # 连接两个子节点并将根节点维度转为out_channel
#         if levels == 1:
#             self.root = Root(root_dim, out_channels, root_kernel_size,
#                              root_residual)
#         self.level_root = level_root
#         self.root_dim = root_dim
#         self.downsample = None
#         self.project = None
#         self.levels = levels
#         #level2-5的stride=2>1
#         if stride > 1:
#             self.downsample = nn.MaxPool2d(stride, stride=stride)
#         #project就是映射，如果输入输出通道数不同则将输入通道数映射到输出通道数
#         if in_channels != out_channels:
#             self.project = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#             )
#
#     def forward(self, x, residual=None, children=None):
#         children = [] if children is None else children
#         # 这里stride都是2，所以进行降采样
#         #level2-5均有下采样
#         bottom = self.downsample(x) if self.downsample else x
#         # 映射至统一维度，从而进行concate
#         #需要先看是否有映射（目前来看应该都有 输入输出均不同）
#         residual = self.project(bottom) if self.project else bottom
#         # 先下采样，再映射得到residual，residual传入basicblock的前向传播，前向传播的residual不再是None
#         #level2 level_root=false 其余的level3-5均是True
#         #level2的时候好像是没有根节点
#         if self.level_root:
#             children.append(bottom)
#
#         x1 = self.tree1(x, residual)
#         if self.levels == 1:
#             x2 = self.tree2(x1)
#             # *children？？？？？？
#             # level2中children=[]
#             x = self.root(x2, x1, *children)
#         else:
#             children.append(x1)
#             x = self.tree2(x1, children=children)
#         # 需要append的数据有：bottom, x1，分别是初始输入，左子树输出，一共加起来的channels=in_channels + out_channels * 2
#         return x
#
#
# # DLA([1, 1, 1, 2, 2, 1],[16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
# # 这部分负责将Tree按照顺序连接起来，在forward()中定义
# # 主要讲的是模块间的特征融合，如类似与ResNet的残差连接。
# class DLA(nn.Module):
#     #DLA([1, 1, 1, 2, 2, 1],
#     #    [16, 32, 64, 128, 256, 512],
#     #    block=BasicBlock, **kwargs)
#     def __init__(self, levels, channels, attention_module, num_classes=1000,
#                  block=BasicBlock, residual_root=False, linear_root=False):
#         super(DLA, self).__init__()
#         self.channels = channels
#         self.num_classes = num_classes
#         # 16→16 conv+bn+relu
#         self.base_layer = nn.Sequential(
#             #3 16 7x7 1 3
#
#             nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
#                       padding=3, bias=False),
#             nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True))
#         # level0,1作为前面两个小方块
#         # 16→16 conv+bn+relu
#         #16 16 1
#         self.level0 = self._make_conv_level(
#             channels[0], channels[0], levels[0])
#         # 16→32 conv+bn+relu
#         #16 32 1 2
#         self.level1 = self._make_conv_level(
#             channels[0], channels[1], levels[1], stride=2)
#         # 32→64 一层树
#         # tree是每个红色框，是树根
#         #1 BasicBlock 32 64 2 false false
#         self.level2 = Tree(levels[2], block, channels[1], channels[2], stride=2,
#                            level_root=False,
#                            root_residual=residual_root)
#         # 64→128 两层树
#         #2 BasicBlock 64 128 2 True false
#         self.level3 = Tree(levels[3], block, channels[2], channels[3], stride=2,
#                            level_root=True, root_residual=residual_root)
#         # 128→256 两层树
#         #2 BasicBlock 128 256 2 True false
#         self.level4 = Tree(levels[4], block, channels[3], channels[4], stride=2,
#                            level_root=True, root_residual=residual_root)
#         # 256→512 一层树
#         #1 BasicBlock 256 512 2 True false
#         self.level5 = Tree(levels[5], block, channels[4], channels[5], stride=2,
#                            level_root=True, root_residual=residual_root)
#
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()
#
#     def _make_level(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.MaxPool2d(stride, stride=stride),
#                 nn.Conv2d(inplanes, planes,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#             )
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample=downsample))
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
#         modules = []
#         for i in range(convs):
#             modules.extend([
#                 nn.Conv2d(inplanes, planes, kernel_size=3,
#                           stride=stride if i == 0 else 1,
#                           padding=dilation, bias=False, dilation=dilation),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#                 nn.ReLU(inplace=True)])
#             inplanes = planes
#         return nn.Sequential(*modules)
#
#     def forward(self, x):
#         y = []
#         #基础卷积层
#         x = self.base_layer(x)
#         for i in range(6):
#             #拼接6个level_layer,level(0-5)
#             # getattr(alex,‘age’) 和 alex.age都能获取alex对象的age属性的值
#             x = getattr(self, 'level{}'.format(i))(x)
#             y.append(x)
#         # 返回的y中包含6个level的结果
#         return y
#
#     def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
#         # fc = self.fc
#         if name.endswith('.pth'):
#             model_weights = torch.load(data + name)
#         else:
#             model_url = get_model_url(data, name, hash)
#             model_weights = model_zoo.load_url(model_url)
#         num_classes = len(model_weights[list(model_weights.keys())[-1]])
#         self.fc = nn.Conv2d(
#             self.channels[-1], num_classes,
#             kernel_size=1, stride=1, padding=0, bias=True)
#         self.load_state_dict(model_weights)
#         # self.fc = fc
#
#
#
# def dla34(pretrained=True,attention_type="simam", **kwargs):  # DLA-34
#     ######################################################
#     # 新添加的注意力机制
#     attention_module = get_attention_module(attention_type)
#
#     if attention_type == "se" or attention_type == "cbam":
#         attention_module = functools.partial(attention_module, reduction=4)
#     elif attention_type == "simam":
#         attention_module = functools.partial(attention_module, e_lambda=1e-5)
#
#     #######################################################
#     b = lambda in_planes, planes, stride, dilation: \
#         BasicBlock(in_planes, planes, stride, dilation, attention_module=attention_module)
#
#     model = DLA([1, 1, 1, 2, 2, 1],
#                 [16, 32, 64, 128, 256, 512], attention_module=attention_module,
#                 block=b, **kwargs)
#     if pretrained:
#         model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
#     return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


# Dcn=dcn→bn→relu
class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        #self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deform_groups=1)
    def forward(self, x):
        x = self.conv(x)
        #激活函数
        x = self.actf(x)
        return x


# 负责上采样并在DeformConv中加入可变形卷积提高卷积核的视野范围、根据任务优化卷积核形状
# 主要讲的是不同层级间的特征融合，如DenseNet的Connection连接。
class IDAUp(nn.Module):
    '''
    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j])
    ida(layers, len(layers) -i - 2, len(layers))
    '''
    # 256 [256 512] [1 2]
    # o表示out_channels
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        #len(channels)=2 i=1~2
        # i=2时通道索引怎么办？？？
        # range()左闭右开，第一次dlaup时range(1,2),只有1！！！！
        for i in range(1, len(channels)):
            #c=512
            c = channels[i]
            #f=2
            f = int(up_f[i])
            #conv2d(512,27,(3,3),stride=(1,1),padding=(1,1))
            #归一化 激活函数
            # 使用dcn上采样，通道从c变为o
            proj = DeformConv(c, o)
            # conv2d(256,27,(3,3),stride=(1,1),padding=(1,1))
            # 归一化 激活函数
            # 通道不变的dcn
            # node = DeformConv(o, o)
            node = Partial_conv3(o, 4, 'split_cat')
            # node = MLPBlock(dim=o,
            #                 n_div=4,
            #                 mlp_ratio=2.0,
            #                 drop_path=0,
            #                 layer_scale_init_value=0,
            #                 norm_layer=nn.BatchNorm2d,
            #                 act_layer=partial(nn.ReLU, inplace=True),
            #                 pconv_fw_type='split_cat')
            # 反卷积
            #ConvTranspose2d(256,256,(4,4),(2,2),(1,1),256)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            #填充上采样权重
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        # dcn调用的：4,3,2-6
        # 每个i：变通道dcn→upsample→与i-1的残差连接→通道不变dcn
        # dlaseg调用的：0-3，则i=1,2
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            # layers5即为levels6的输出！！！
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    # first_level=2
    # channels[self.first_level:]=64 128 256 512
    # 1 2 4 8
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        # for i在0-3内，设置ida_0,ida_1,ida_2
        for i in range(len(channels) - 1):
            # numbers[-5:-2]表示从列表中的倒数第五个元素（包括）到倒数第二个元素（不包括）的子列表。
            #第一次循环j=-2 选取channels最后两个
            j = -i - 2
            #channels[j]=256
            #in_channels[j:]=256 512
            #scales[j:] // scales[j]=[4 8]//4=[1 2]
            # setattr() 是 Python 内置函数之一，用于设置对象的属性或方法。
            # setattr() 接受三个参数：第一个参数是对象，第二个参数是字符串形式的属性名或方法名，第三个参数是要设置的属性或方法的值或引用。
            # ida_0:
            # j=-2  channels[j]=256    in_channels[j:]=[256,512]    scales[j:]//scales[j]=[4,8]//4=[1,2]
            # ida_1:
            # j=-3  channels[j]=128    in_channels[j:]=[128,256,256]    scales[j:]//scales[j]=[2,4,4]//2=[1,2,2]
            # ida_2:
            # j=-4  channels[j]=64    in_channels[j:]=[64,128,128,128]  scales[j:]//scales[j]=[1,2,2,2]//1=[1,2,2,2]
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            # j=-2 scales[-1:] = scales[-2],即 1 2 4 8 变为 1 2 4 4
            # j=-3 scales[-2:] = scales[-3],即 1 2 4 8 变为 1 2 2 2
            # j=-4 scales[-3:] = scales[-4],即 1 2 4 8 变为 1 1 1 1
            #1,2,4,8->1,2,4,4->1 2 2 2->1 1 1 1
            scales[j + 1:] = scales[j]
            # 如果在循环体中不需要用到自定义变量，可将自定义变量写为下划线‘_’
            # j=-2 in_channels[-1:] = channels[-2]
            #64 128 256 512->64 128 256 256->64 128 128 128->64 64 64 64
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        # layers即传入的x，out使用x的最后一位，主干网络的最后一个通道
        out = [layers[-1]] # start with 32
        # len(layers) = 6 ，即6层树的输出
        # startp = first_level = 2
        # i从0-2！！！！
        for i in range(len(layers) - self.startp - 1):
        # for i in range(6 - self.startp - 1):
            # getattr() 是 Python 内置函数之一，用于获取对象的属性或方法。
            # getattr() 接受两个参数：第一个参数是对象，第二个参数是字符串形式的属性名或方法名。
            ida = getattr(self, 'ida_{}'.format(i))
            # 也就是每次调用dlaup相当于用三次idaup？
            # layers代表x，第二个参数三次分别是4,3,2，第三个参数是6
            # ida(layers, 6 -i - 2, 6)
            # print(layers[5].size())
            # print('_____')
            # print(layers[4].size())
            # print('_____')
            # print(layers[3].size())
            # print('_____')
            # print(layers[2].size())
            # print('_____')
            # print(layers[1].size())
            # print('_____')
            # print(layers[0].size())
            ida(layers, len(layers) - i - 2, len(layers))
            # list.insert(index, obj)参数：
            # 　　index -- 对象obj需要插入的索引位置。
            # 　　obj -- 要插入列表中的对象。
            # 在out的开头插入经过idaup后的layers的最后一个通道
            out.insert(0, layers[-1])
        # out列表又有四个，分别是三次idaup的layers最后通道，二次，一次，原始最后通道
        # 即通道为64 128 256 512
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class FasterNet(nn.Module):

    def __init__(self,
                 # PatchEmbed的输入通道维度
                 in_chans=3,
                 num_classes=1000,
                 # PatchEmbed的输出通道维度
                 embed_dim=96,
                 # basic block的深度
                 depths=(1, 2, 8, 2),
                 # mlp内部的卷积通道膨胀倍数
                 mlp_ratio=2.,
                 # 部分卷积比例
                 n_div=4,
                 # 用于PatchEmbed的
                 patch_size=4,
                 patch_stride=2,
                 # 用于PatchMerging的
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 # PatchEmbed是否加bn层
                 patch_norm=True,
                 # 特征头通道？
                 feature_dim=1280,
                 # drop out的比例？
                 drop_path_rate=0.1,
                 # 初始化可学习权重的比例？
                 layer_scale_init_value=0,
                 # 正则化层
                 norm_layer='BN',
                 # 激活函数
                 act_layer='RELU',
                 # det头用的
                 fork_feat=False,
                 # 初始化权重
                 init_cfg=None,
                 # 预训练权重
                 pretrained=None,
                 # 推理时将pconv_fw_type设置为slicing，加快推理速度？
                 pconv_fw_type='split_cat',
                 attention_type=None,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        # 一共四个stage
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # 好像用不上？
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.attention_type = attention_type
        self.gate_fn = nn.Sigmoid()

        # self.base_layer = nn.Sequential(
        #     #3 16 7x7 1 3
        #
        #     nn.Conv2d(3, 16, kernel_size=7, stride=1,
        #               padding=3, bias=False),
        #     nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True))

        # split image into non-overlapping patches
        # 特征嵌入层，通道由in_chans变为embed_dim,3→96
        # TODO：改成16→32
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # 随机深度衰减？
        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        # 共四次迭代，i=0,1,2,3
        # 每次迭代，通道翻倍，故从96（原始）→192（i=0结束）→384（i=1结束）→768（i=2结束）→768（i=3时不merging）
        # TODO：32(原)→64→128→256→512，最后一次仍然需要通道翻倍
        for i_stage in range(self.num_stages):
            layer = []
            fasterblock = []
            pm = []
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)
            layer.append(stage)
            fasterblock.append(stage)

            # patch merging layer
            # 最后一次迭代没有merging层
            # if i_stage < self.num_stages - 1:
                # stages_list.append(
                #     PatchMerging(patch_size2=patch_size2,
                #                  patch_stride2=patch_stride2,
                #                  dim=int(embed_dim * 2 ** i_stage),
                #                  norm_layer=norm_layer)
                # )
            stage2 = PatchMerging(patch_size2=patch_size2,
                                  patch_stride2=patch_stride2,
                                  dim=int(embed_dim * 2 ** i_stage),
                                  norm_layer=norm_layer)
            stages_list.append(stage2)
            layer.append(stage2)
            pm.append(stage2)

            layers = nn.Sequential(*layer)
            fasterblock = nn.Sequential(*fasterblock)
            pm = nn.Sequential(*pm)

            setattr(self, 'layer_{}'.format(i_stage), layers)
            setattr(self, 'fasterblock_{}'.format(i_stage), fasterblock)
            setattr(self, 'pm_{}'.format(i_stage), pm)

            inp = int(embed_dim * 2 ** i_stage)
            # oup = 2*inp
            oup = inp
            short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=1 // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
            setattr(self, 'short_{}'.format(i_stage), short_conv)

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        # if self.fork_feat:
        #     self.forward = self.forward_det
        #     # add a norm layer for each output
        #     self.out_indices = [0, 2, 4, 6]
        #     for i_emb, i_layer in enumerate(self.out_indices):
        #         if i_emb == 0 and os.environ.get('FORK_LAST3', None):
        #             raise NotImplementedError
        #         else:
        #             layer = norm_layer(int(embed_dim * 2 ** i_emb))
        #         layer_name = f'norm{i_layer}'
        #         self.add_module(layer_name, layer)
        # else:
        #     self.forward = self.forward_cls
        #     # Classifier head
        #     self.avgpool_pre_head = nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
        #         act_layer()
        #     )
        #     self.head = nn.Linear(feature_dim, num_classes) \
        #         if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward(self, x: Tensor) -> Tensor:
        # print("++", x.device, self.unused_p.device, self.patch_embed.proj.weight.device)
        y = []
        y.append(x)
        # x = self.base_layer(x)
        x = self.patch_embed(x)
        # x = self.stages(x)
        y.append(x)
        if self.attention_type == 'dfc':
            for i in range(self.num_stages):
                # 4层
                # getattr(alex,‘age’) 和 alex.age都能获取alex对象的age属性的值
                x1 = getattr(self, 'fasterblock_{}'.format(i))(x)
                x2 = getattr(self, 'short_{}'.format(i))(F.avg_pool2d(x, kernel_size=2, stride=2))
                x = x1 * F.interpolate(self.gate_fn(x2), size=(x1.shape[-2], x1.shape[-1]), mode='nearest')
                x = getattr(self, 'pm_{}'.format(i))(x)
                y.append(x)
        else:
            for i in range(self.num_stages):
                # 4层
                # getattr(alex,‘age’) 和 alex.age都能获取alex对象的age属性的值
                x = getattr(self, 'layer_{}'.format(i))(x)
                y.append(x)
        # 返回的y中包含6个level的结果
        return y

class DLASeg(nn.Module):
    #dla34 heads={hm:1,wh:4 id:128 reg:2} false 4 1 5 256
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv,attention_type, out_channel=0):
        # base_name是dla
        super(DLASeg, self).__init__()
        #如果下采样倍数不对 报错
        assert down_ratio in [2, 4, 8, 16]
        # down_ratio = 4,first_level = 2
        #取对数？？2的n次方 1 2 3 4
        self.first_level = int(np.log2(down_ratio))
        # last_level = 5
        #5级
        self.last_level = last_level
        #self.base = dla34(pretrained=pretrained)
        ###差点看吐的神经网络DLA34
        # 加载DLA34作为self.base，gloabal()这里是引用外部dla34()函数初始化并返回DLA网络
        # self.base = globals()[base_name](pretrained=pretrained,attention_type=attention_type)
        if base_name == 't0relu':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(1, 2, 8, 2),
                drop_path_rate=0.,
                act_layer='RELU',
                fork_feat=True,attention_type=attention_type
            )
        elif base_name == 't0':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(1, 2, 8, 2),
                drop_path_rate=0.,
                act_layer='GELU',
                fork_feat=True,attention_type=attention_type
            )
        elif base_name == 't1':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(1, 2, 8, 2),
                drop_path_rate=0.02,
                act_layer='GELU',
                fork_feat=True,attention_type=attention_type
            )
        elif base_name == 't2':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(1, 2, 8, 2),
                drop_path_rate=0.05,
                act_layer='RELU',
                fork_feat=True,attention_type=attention_type
            )
        elif base_name == 's':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(1, 2, 13, 2),
                drop_path_rate=0.15,
                act_layer='RELU',
                fork_feat=True,attention_type=attention_type
            )
        elif base_name == 'm':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(3, 4, 18, 3),
                drop_path_rate=0.2,
                act_layer='RELU',
                fork_feat=True,attention_type=attention_type
            )
        elif base_name == 'l':
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(3, 4, 18, 3),
                drop_path_rate=0.3,
                act_layer='RELU',
                fork_feat=True,attention_type=attention_type
            )
        else:
            self.base = FasterNet(
                mlp_ratio=2.0,
                embed_dim=32,
                depths=(1, 2, 4, 2),
                drop_path_rate=0.,
                act_layer='GELU',
                fork_feat=True, attention_type=attention_type
            )
        # Fasternet_t0
        # Fasternet_t
        # self.base = FasterNet(
        # mlp_ratio=2.0,
        # embed_dim=32,
        # depths=(1, 2, 8, 2),
        # drop_path_rate=0.,
        # act_layer='RELU',
        # fork_feat=True
        # )
        # channels是每个根部的Root向量维度，最后输出为512channels
        #16 32 64 128 256 512数组
        # channels = self.base.channels
        channels = [16, 32, 64, 128, 256, 512]
        #scales = 1 2 4 8
        # **幂运算 2的0,1,2,3次方
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        #first_level=2
        #channels[self.first_level:]=64 128 256 512
        #上采样  通道变回原来的 图像大小变回原来的
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]
        #上采样?
        # out_channel=64
        # channels[self.first_level:self.last_level]=64 128 256 512
        # 2 ** i for i in range(self.last_level - self.first_level)=1 2 4 8
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])

        ######################################################
        #在此处加新网络
        #self.newnetwork
        ######################################################

        # heads = {hm: 1, wh: 4 id: 128 reg: 2}
        # head_conv = 256
        # 定义每个head的网络结构
        #头节点
        self.heads = heads
        for head in self.heads:
            # channel大小
            #分类1 4 128 2
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  #conv 64->256
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  #conv 256->1
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                  #添加了偏置？？？？？？？
                fc[-1].bias.data.fill_(-2.19)
              else:
                  #添加fc权重
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        # print(x.device, self.unused_p.device, self.base.unused_p.device, self.base.patch_embed.proj.weight.device)
        # base是dla34
        x = self.base(x)
        # x有6维，依次是六个层的输出，16 32 64 128 256 512
        x = self.dla_up(x)
        # x有四维，分别是：对最后一个维度进行（三次idaup，两次，一次，原始x）
        # 64 128 256 512

        y = []
        # i=0-3
        # 将x按顺序克隆给y，不要最后一个
        # range(3)即0,1,2
        for i in range(self.last_level - self.first_level):
            #创建x[i]的克隆副本送给y
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        # y为经过ida后的输出，共三维，需要第三个，即y[-1]

        z = {}
        for head in self.heads:
            # y[-1]为y最后的输出
            z[head] = self.__getattr__(head)(y[-1])
        return [z]

# model.py中的create调用该函数创建模型
#num_layers=34  heads={hm:1,wh:4 id:128 reg:2} head_conv=256
def get_pose_net(num_layers, heads, attention_type, head_conv=256, down_ratio=4):
  model = DLASeg(num_layers, heads,
                 pretrained=False,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv,
                 attention_type=attention_type
                 )
  return model

