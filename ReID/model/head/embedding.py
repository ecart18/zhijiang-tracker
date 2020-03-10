from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):

    def __init__(self, backbone, cut_at_pooling=False, num_features=1024,
                 norm=False, dropout=0, num_classes=128):
        super(Embedding, self).__init__()

        self.cut_at_pooling = cut_at_pooling
        self.backbone = backbone

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.backbone.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                nn.init.kaiming_normal_(self.feat.weight, mode='fan_out')
                nn.init.constant_(self.feat.bias, 0)
                nn.init.constant_(self.feat_bn.weight, 1)
                nn.init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                nn.init.normal_(self.classifier.weight, std=0.001)
                nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        for name, module in self.backbone._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x
