import numpy as np


def filter_person(det):

    def bbox_surrounded(person_bbox, bbox):
        # (x1, y1, w, h)
        if (person_bbox[0] > bbox[0]) and (person_bbox[1] > bbox[1]) and \
            ((person_bbox[0] + person_bbox[2]) < (bbox[0] + bbox[2])) and \
                ((person_bbox[1] + person_bbox[3]) < (bbox[1] + bbox[3])):
            return True
        else:
            return False

    def bbox_aspect_ratio(person_bbox):
        if person_bbox[3] / person_bbox[2] < 2.:
            return True
        else:
            return False

    def left_distance(person_bbox, bbox):
        if abs(bbox[0] - person_bbox[0]) > 0.3*bbox[3]:
            return True
        else:
            return False
    if det.ndim == 1:
        det = np.expand_dims(det, axis=0)

    if det.shape[0] > 1:
        person_det = det[det[:, 7] == 0, :]
        delete = []
        for det_idx in range(len(det)):
            cls = int(det[det_idx, 7])
            if cls == 5:  # for bus
                bbox = det[det_idx, 2:6]
                for person_idx in range(len(person_det)):
                    person_bbox = person_det[person_idx, 2:6]
                    if bbox_surrounded(person_bbox, bbox):
                        if bbox_aspect_ratio(person_bbox):
                            delete.append(person_idx)
                        if left_distance(person_bbox, bbox):
                            delete.append(person_idx)
        if len(delete) > 0:
            delete = np.unique(sorted(delete)).tolist()
            person_det = np.delete(person_det, delete, axis=0)
        return person_det
    else:
        return det

