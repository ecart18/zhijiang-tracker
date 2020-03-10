
import torch
import numpy as np
from H4tracker.utils.bbox import overlap_ratio


class Node:
    """
    The Node is the basic element of a track. it contains the following information:
    1) frame index (frame index)
    2) box (a box (l, t, r, b)
    3) id, detection id for recorder indexing
    """

    def __init__(self, frame_index, id, bbox):
        self.frame_index = frame_index
        self._id = id
        self.bbox = bbox

    def get_box(self, frame_index, recoder):
        if frame_index - self.frame_index >= 4000:
            return None
        return recoder.all_boxes[self.frame_index][self._id, :]

    def get_iou(self, frame_index, recoder, box_id):
        return recoder.all_iou[frame_index][self.frame_index][box_id, self._id]


class Track:
    """
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes
    2) track id. it is unique it identify each track
    3) track pool id. it is a number to give a new id to a new track
    4) age. age indicates how old is the track
    5) max_age. indicates the dead age of this track
    """

    _id_pool = 0

    def __init__(self, init_node, init_feature):
        self._nodes = list()
        self._nodes.append(init_node)

        self._features = list()
        self._features.append(init_feature)

        self._age = 1
        self._missed_frame = 0
        self._missed_detection_score = np.array(1e5)

        self._id = Track._id_pool
        Track._id_pool += 1

        self._valid = True   # indicate this track is merged

    def __del__(self):
        for n in self._nodes:
          del n

    def get_track_id(self):
        return self._id

    def set_track_id(self, track_id):
        self._id = track_id

    def get_tracklet_nodes(self):
        return self._nodes

    def update_track(self, feame_index=None, det_id=None, bbox=None,
                     embedding_feature=None, visual_mode=False):
        if visual_mode:
            node = Node(feame_index, det_id, bbox=bbox)
            self._nodes.append(node)
            return True
        if bbox is not None:
            self._add_new_node(feame_index, det_id, bbox, embedding_feature)
            return True
        else:
            self._missed_frame += 1

    def _add_new_node(self, feame_index, det_id, bbox, embedding_feature):
        node = Node(feame_index, det_id, bbox=bbox)
        self._age += 1
        self._missed_frame = 0
        self._nodes.append(node)
        self._features.append(embedding_feature)

    def get_distance(self, frame_index, detections, features):
        pick = []
        dis = np.zeros(len(detections)) + self._missed_detection_score
        for idx in range(len(detections)):
            bbox = detections[idx, :]
            if self._gate(frame_index, bbox):
                pick.append(idx)
        if len(pick) > 0:
            features = features[pick, :]
            distance = self._get_distance(torch.stack(self._features, dim=0), features)
            for idx, val in enumerate(pick):
                dis[val] = distance[idx]
        return dis

    def get_distance_latergating(self, frame_index, detections, features):
        distance = self._get_distance(torch.stack(self._features, dim=0), features)
        dis_th = 0.10
        # area = self._area(history=3)
        for idx in range(len(detections)):
            if not self._gate(frame_index, bbox=detections[idx, :]):
                if distance[idx] > dis_th:
                    distance[idx] = self._missed_detection_score
        return distance

    def _gate(self, frame_index, bbox, history=1):  # TODO No gating for exceeding 20 frames
        if len(self._nodes) > 0:
            node = self._nodes[-history:]
            history_bbox = np.array([n.bbox for n in node])
            history_delta_frame = [frame_index - n.frame_index for n in node]
            history_iou = overlap_ratio(history_bbox, bbox)
            index = np.argmax(history_iou)
            delta_frame = history_delta_frame[index]
            iou = float(history_iou[index])
            if delta_frame in range(0, 41):
                if iou < pow(0.3, delta_frame):
                    return False
            else:
                return False
        return True

    def verify(self, frame_index, recorder, box_id):
        for n in self._nodes:
            delta_f = frame_index - n.frame_index
            if delta_f in range(0, 41):
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou is None: continue
                if iou < pow(0.3, delta_f):
                    return False
        return True

    def get_track_missed_frame(self):
        return self._missed_frame

    @staticmethod
    def _get_distance(tracklet, detections):
        num_tracklet= tracklet.size(0)
        num_detection = detections.size(0)
        # Compute pairwise distance, replace by the official when merged
        t_sum_sqr = torch.pow(tracklet, 2).sum(dim=1, keepdim=True).expand(num_tracklet, num_detection)
        d_sum_sqr = torch.pow(detections, 2).sum(dim=1, keepdim=True).expand(num_detection, num_tracklet)
        dist = t_sum_sqr + d_sum_sqr.t()
        dist.addmm_(1, -2, tracklet, detections.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.min(dim=0)[0].cpu().numpy()


class Tracks:
    """
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous tracks
    """
    def __init__(self, max_draw_track_node=20):
        self.tracks = list()  # the set of tracks
        self.saved_tracks = list()
        self.max_drawing_track = max_draw_track_node

    def __getitem__(self, item):
        return self.tracks[item]

    def add_new_track(self, feame_index, det_id, bbox, embedding_feature, track_id=None, image=None):
        node = Node(feame_index, det_id, bbox=bbox)
        t = Track(init_node=node, init_feature=embedding_feature)
        if track_id is not None:
            t.set_track_id(track_id)
        self.tracks.append(t)

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.get_track_id() == id:
                return t
        return None

    def get_all_tracks(self, minimal_len=10):
        output_tracks = [t for t in self.saved_tracks if len(t.get_tracklet_nodes()) >= minimal_len]
        return self.tracks + output_tracks

    def volatile_tracks(self):  # Todo acclerate
        if len(self.tracks) > 5000:
            # start to delete the most oldest tracks
            all_missed_frames = [t.cal_track_missed_frame() for t in self.tracks]
            oldest_track_index = np.argmax(all_missed_frames)
            del self.tracks[oldest_track_index]

    def one_frame_pass(self):
        keep_track_set = list()
        saved_track_set = list()
        for i, t in enumerate(self.tracks):
            if t.get_track_missed_frame() > 100:  #TODO
                saved_track_set.append(i)
                continue
            keep_track_set.append(i)
        self.saved_tracks = self.saved_tracks + [self.tracks[i] for i in saved_track_set]
        self.tracks = [self.tracks[i] for i in keep_track_set]

