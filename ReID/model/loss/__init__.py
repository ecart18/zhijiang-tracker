from __future__ import absolute_import

from .loss import TripletLoss

__all__ = [
    'TripletLoss',
    'build_loss'
]


__LOSS = {
    'TripletLoss': TripletLoss,
}


def build_loss(name='TripletLoss', *args, **kwargs):
    if name not in __LOSS:
        raise KeyError("Unknown loss:", name)
    return __LOSS[name](*args, **kwargs)

