
from __future__ import absolute_import

import numpy as np




def overlap_ratio(rect1, rect2):
    """
    Compute overlap ratio between two rects
        - rect: 1d array of [x,y,x,y] or
            2d array of N x [x,y,x,y]
    """
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 2], rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 3], rect2[:, 3])
    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = (rect1[:, 2] - rect1[:, 0]) * (rect1[:, 3] - rect1[:, 1]) + \
            (rect2[:, 2] - rect2[:, 0]) * (rect2[:, 3] - rect2[:, 1]) - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

        Original code from [1]_ has been adapted to include confidence score.

        .. [1] http://www.pyimagesearch.com/2015/02/16/
            faster-non-maximum-suppression-python/

        Examples
        --------
        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

        Parameters
        ----------
        boxes : ndarray
            Array of ROIs (x, y, width, height).
        max_bbox_overlap : float
            ROIs that overlap more than this values are suppressed.
        scores : Optional[array_like]
            Detector confidence score.

        Returns
        -------
        List[int]
            Returns indices of detections that have survived non-maxima suppression.
    """
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)
        # idxs = np.arange(len(boxes))[::-1]
        # idxs = np.arange(len(boxes))

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
