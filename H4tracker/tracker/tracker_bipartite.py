
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracks import Track, Tracks
from H4tracker.utils.recorder import FeatureRecorder
from H4tracker.utils.infer import InferenceUtil


class BiTracker:
    def __init__(self, affinity_model=None,
                 minimal_tracklet_len=10, roi_verify_max_iteration=2,
                 roi_verify_punish_rate=0.6, pred_nms_max_overlap=0.9):

        self.minimal_tracklet_len = minimal_tracklet_len
        self.roi_verify_max_iteration = roi_verify_max_iteration
        self.roi_verify_punish_rate = roi_verify_punish_rate
        self.pred_nms_max_overlap = pred_nms_max_overlap

        self.affinity_model = affinity_model

        self.tracks = Tracks()
        self.feature_recorder = FeatureRecorder()

        Track._id_pool = 0
        self.frame_index = 0
        self.predict_detections = []

    def get_all_tracks(self):
        return self.tracks.get_all_tracks(minimal_len=self.minimal_tracklet_len)

    def _assinment_heuristic(self, tracks, distance, ids, detections_id=None):
        # find the corresponding by the distance matrix
        track_num = distance.shape[0]
        if track_num > 0:
            box_num = distance.shape[1]
        else:
            box_num = 0
        if detections_id is None:
            detections_id = list(range(box_num))

        row_index, col_index = linear_sum_assignment(distance)

        # verification by iou
        verify_iteration = 0
        while verify_iteration < self.roi_verify_max_iteration:
            is_change = False
            for idx, track_idx in enumerate(row_index):
                track_id = ids[track_idx]
                det_idx = col_index[idx]
                det_id = detections_id[det_idx]
                t = tracks.get_track_by_id(track_id)
                if not t.verify(self.frame_index, self.feature_recorder, det_id):
                    distance[track_idx, det_idx] *= self.roi_verify_punish_rate
                    is_change = True
            if is_change:
                row_index, col_index = linear_sum_assignment(distance)
            else:
                break
            verify_iteration += 1

        assigned_pairs_id = []
        for idx, track_idx in enumerate(row_index):
            track_id = ids[track_idx]
            det_idx = col_index[idx]
            det_id = detections_id[det_idx]
            if distance[track_idx, det_idx] < 1e4:  #TODO it is need to remove the out of gating pair
                assigned_pairs_id.append((track_id, det_id))

        unassigned_tracks_id = sorted(list(set(ids) - set([pairs[0] for pairs in assigned_pairs_id])))
        unassigned_detections_id = sorted(list(set(detections_id) - set([pairs[1] for pairs in assigned_pairs_id])))

        return assigned_pairs_id, unassigned_tracks_id, unassigned_detections_id

    def update_detections(self, image, detections):
        patchs, patchs_bbox, img_size = InferenceUtil.convert_image2patch(image, detections)  # [x1, y1, x2, y2]
        return patchs, patchs_bbox

    def update_tracker(self, frame_index, image, detections, force_init=False, total_frame=None):
        """
        :param frame_index: frame number of video
        :param image: original image for this frame
        :param detections: all detections results for one frame
        :param force_init: force this frame as 1st frame
        :return:
        """

        self.frame_index = frame_index
        self.img_sz = image.shape[0:2]  # [height, width]

        patchs, detections = self.update_detections(image, detections)
        if patchs is None:
            return image

        with torch.no_grad():
            features = self.affinity_model(torch.Tensor(patchs).cuda())
            if isinstance(features, tuple):
                features = torch.cat(features[0:2], dim=1)
        self.feature_recorder.update(self.frame_index, features.data, detections)

        # Create new branch from the every detection results
        if self.frame_index == 0 or force_init or len(self.tracks.tracks) == 0:
            for index in range(detections.shape[0]):
                self.tracks.add_new_track(feame_index=self.frame_index,
                                          det_id=index,
                                          bbox=detections[index, :],
                                          embedding_feature=self.feature_recorder.get_feature(self.frame_index, index),
                                          image=image)
            self.tracks.one_frame_pass()
            return image

        else:
            distance = []
            ids = []
            for track in self.tracks.tracks:
                ids.append(track.get_track_id())
                features = self.feature_recorder.get_feature(self.frame_index, None)
                dis = track.get_distance_latergating(self.frame_index, detections, features)
                distance.append(dis)
            distance = np.array(distance).squeeze().reshape(-1, len(detections))
            if len(distance) > 0:
                assigned_pairs_id, unassigned_tracks_id, unassigned_detections_id = \
                  self._assinment_heuristic(self.tracks, distance, ids)

                # update the tracks
                for track_id, det_id in assigned_pairs_id:
                    track = self.tracks.get_track_by_id(track_id)
                    track.update_track(feame_index=self.frame_index,
                                       det_id=det_id,
                                       bbox=self.feature_recorder.get_box(self.frame_index, det_id),
                                       embedding_feature=self.feature_recorder.get_feature(self.frame_index, det_id),
                                       visual_mode=False)

                for track_id in unassigned_tracks_id:
                    track = self.tracks.get_track_by_id(track_id)
                    track.update_track(feame_index=self.frame_index,
                                       det_id=None, bbox=None, embedding_feature=None,visual_mode=False)

                # add new track
                for det_id in unassigned_detections_id:
                    self.tracks.add_new_track(feame_index=self.frame_index,
                                              det_id=det_id,
                                              bbox=self.feature_recorder.get_box(self.frame_index, det_id),
                                              embedding_feature=self.feature_recorder.get_feature(self.frame_index, det_id))

            # remove the old track
            self.tracks.one_frame_pass()

        return image

