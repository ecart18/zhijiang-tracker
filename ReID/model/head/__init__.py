from __future__ import absolute_import

from .embedding import Embedding

__all__ = [
    'Embedding',
    'get_head'
]

__HEAD = {
    'embedding': Embedding,
}


def get_head(backbone, name='embedding', **kwargs):
    return __HEAD[name](backbone, **kwargs)

