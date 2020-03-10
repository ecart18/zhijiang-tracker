from __future__ import absolute_import

import copy
import numpy as np
from itertools import groupby
from operator import itemgetter
from .bbox import non_max_suppression


def nms_interpolate(result, nms_max_overlap=0.9, max_missed_fram=20):
    result[:, 6] = 1
    result = interpolate_tracklet(result, max_missed_fram=max_missed_fram)
    frame = result[:, 0].tolist()  # object id
    frame_set = sorted(list(set(frame)))
    result_nms = np.array([])
    for id_x in frame_set:
        res = result[result[:, 0] == id_x, :]
        boxes = res[:, 2:6]
        scores = res[:, 6]
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        indices = sorted(indices)
        res = res[indices, :]
        res[:, 6] = -1
        if len(result_nms) == 0:
            result_nms = res
        else:
            result_nms = np.concatenate((result_nms, res), axis=0)
    res_sort = result_nms[np.lexsort((result_nms[:, 1], result_nms[:, 0]))]
    return res_sort


def interpolate_tracklet(result, max_missed_fram=10):

    id = result[:, 1].tolist()  # object id
    id_set = sorted(list(set(id)))
    result_inter = np.array([])
    for id_x in id_set:
        blank_line = []
        tracklet = result[result[:, 1] == id_x, :]
        frame_id = list(map(int, tracklet[:, 0].tolist()))
        blank = set(range(frame_id[0], frame_id[-1] + 1)) - set(frame_id)
        blank = sorted(list(blank))
        for k, g in groupby(enumerate(blank), lambda i_x: i_x[0] - i_x[1]):
            blank_line.append(list(map(itemgetter(1), g)))

        # linear interpolate
        res = copy.deepcopy(tracklet)
        for blank in blank_line:
            if (blank[0] <= 1):
                continue
            if len(blank) >= max_missed_fram:
                continue
            start = blank[0] - 1
            end = blank[-1] + 1
            len_blank = len(blank)
            num = len(blank) + 2
            frame = np.array(blank).reshape(-1, 1)
            o_id = np.repeat(np.array(res[0, 1]), repeats=len_blank).reshape(-1, 1)
            post = np.repeat(np.expand_dims(np.array([-1, -1, -1, -1]), axis=1),
                             repeats=len_blank, axis=1).reshape(-1, 4)
            x = np.linspace(res[res[:, 0] == start, 2], res[res[:, 0] == end, 2], num=num)[1:-1].reshape(-1, 1)
            y = np.linspace(res[res[:, 0] == start, 3], res[res[:, 0] == end, 3], num=num)[1:-1].reshape(-1, 1)
            w = np.linspace(res[res[:, 0] == start, 4], res[res[:, 0] == end, 4], num=num)[1:-1].reshape(-1, 1)
            h = np.linspace(res[res[:, 0] == start, 5], res[res[:, 0] == end, 5], num=num)[1:-1].reshape(-1, 1)
            interpolation = np.concatenate((frame, o_id, x, y, w, h, post), axis=1)
            index1 = int((np.where(tracklet[:, 0] == start)[0] + 1).tolist()[0])
            index2 = int((np.where(tracklet[:, 0] == end)[0] + 1).tolist()[0])
            tracklet = np.insert(tracklet, range(index1, index2), interpolation, axis=0)
        if len(result_inter) == 0:
            result_inter = tracklet
        else:
            result_inter = np.concatenate((result_inter, tracklet), axis=0)

    return result_inter
