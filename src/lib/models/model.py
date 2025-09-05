from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

#定义多个网络模型
# from .networks.dlav0 import get_pose_net as get_dlav0
# from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
# from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
# from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
# from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
# from .networks.pose_dla_conv import get_pose_net as get_dla_conv
# from .networks.faster_dcn import get_pose_net as get_pose_net_fasternet
# from .networks.faster_dcn_sc import get_pose_net as get_pose_net_fasternet_sc
# from .networks.ghost_dcn import get_pose_net as get_ghost_dcn
# from .networks.faster_ghost import get_pose_net as get_faster_ghost
from .networks.faster_dfc import get_pose_net as get_faster_dfc
# from .networks.faster_afpn import get_pose_net as get_faster_afpn
# from .networks.emo_dcn import get_pose_net as  get_emo_dcn
# from .networks.emo_afpn import get_pose_net as get_emo_adpn

_model_factory = {
  # 'dlav0': get_dlav0, # default DLAup
  # 'dla': get_dla_dcn,
  # 'dlaconv': get_dla_conv,
  # 'resdcn': get_pose_net_dcn,
  # 'resfpndcn': get_pose_net_fpn_dcn,
  # 'hrnet': get_pose_net_hrnet,
  # 'fasternet': get_pose_net_fasternet,
  # 'fsc': get_pose_net_fasternet_sc,
  # 'ghost': get_ghost_dcn,
  # 'fg': get_faster_ghost,
  'fasterdfc': get_faster_dfc,
  # 'fasterafpn': get_faster_afpn,
  # 'emo': get_emo_dcn,
  # 'emo': get_emo_adpn
}

def create_model(arch, heads, attention_type , head_conv):

  #确定输入模型的dla_34 的数字34
  # num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  num_layers = arch[arch.find('_') + 1:] if '_' in arch else 't0'
  #确定dla_34的dla
  arch = arch[:arch.find('_')] if '_' in arch else arch
  #获取模型（调用一下函数而已）
  get_model = _model_factory[arch]
  #真正的调用函数模型
  model = get_model(num_layers=num_layers, heads=heads, attention_type=attention_type, head_conv=head_conv)
  # model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  #gpu->cpu?
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  # print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  # for key in checkpoint.keys():
  #   print('key = {}'.format(key))
  print('loaded {}'.format(model_path))
  if 'state_dict' in checkpoint:
    _state_dict = checkpoint['state_dict']
  elif 'model' in checkpoint:
    _state_dict = checkpoint['model']
  else:
    _state_dict = checkpoint

  state_dict_ = _state_dict
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

