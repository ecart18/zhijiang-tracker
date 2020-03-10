from __future__ import absolute_import

import os.path as osp
from ReID.model.weights_init import init_weights, load_pretrain_state_dict
from .resnet import resnet50


__BACKBONES = {
    'resnet50': resnet50,
}


def get_backbone(name='resnet50', pretrained=True, need_grad=True, model_path='pretrainmodel/resnet50-19c8e357.pth'):

    if name not in __BACKBONES:
        raise KeyError("Unknown backbone structure:", name)
    backbone = __BACKBONES[name]()
    if pretrained:
        backbone = load_pretrain_state_dict(backbone, name=name,
                                            model_path=osp.join('..', model_path))
        if not need_grad:
            for param in backbone.parameters():
                param.requires_grad = False
            print('Gradients of backbone Network {0} are not required. \n'.format(name))
    else:
        backbone = init_weights(backbone)
    return backbone
