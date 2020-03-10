import torch
import torch.nn as nn
import os.path as osp


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model


def load_pretrain_state_dict(model, model_path=None, name='res50'):
    if osp.exists(model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = model.state_dict()
        update_dict = {k: v for k, v in pretrain_dict.items()
                       if k in model_dict.keys()}  # filter out unnecessary keys
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        print('Loaded pre-trained parameters of {0} from: {1}. \n'.format(name, model_path))
    else:
        raise IOError("File not exist: {0} \n".format(model_path))
    return model


