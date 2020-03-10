
from __future__ import absolute_import

from .tracker_bipartite import BiTracker
from .tracks import Node, Track, Tracks

__all__ = [
    'Node',
    'Track',
    'Tracks',
    'BiTracker',
]


__TRACKS = {
          'bitracker': BiTracker,
         }


def build_tracker(name, **kwargs):
    if name not in __TRACKS:
        raise KeyError("Unknown loss:", name)
    return __TRACKS[name](**kwargs)

