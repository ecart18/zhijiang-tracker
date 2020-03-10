from __future__ import absolute_import

import time


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.batch_reset()
        self.loss_recoder = []

    def batch_reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.loss_batch = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.loss_batch.append(val)

    def recoder(self):
        if len(self.loss_batch) is not 0:
            self.loss_recoder.append(self.loss_batch)

