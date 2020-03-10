
import os
import sys
import os.path as osp
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from H4tracker.utils.osutils import mkdir_p, load_params
from H4tracker.utils.meters import Timer
from H4tracker.utils.post_processing import nms_interpolate
from H4tracker.tracker import BiTracker
from H4tracker.datasets import MOTDataReader
from ReID.model.backbone import get_backbone
from ReID.model.head import get_head


def load_model():
    cudnn.benchmark = True
    device = 'cuda'
    backbone = get_backbone(pretrained=False, need_grad=False)
    affinity_model = get_head(backbone)
    affinity_model = load_params(affinity_model, "../model/reid/model_best.pth.tar").to(device).eval()
    return affinity_model


def test(args):
    timer = Timer()

    affinity_model = load_model()

    mot_data_root = "../data/zj_test/level2"
    det_data_root = "../result/level2/det"
    det_file_name_format = os.path.join(det_data_root, 'b' + '{:01}', 'det.txt')
    dataset_image_folder_format = os.path.join(mot_data_root, 'b' + '{:01}/img1')
    saved_file_name_format = os.path.join("../result/level2/trk", 'b' + '{:01}.txt')
    data_index = [1, 2, 3, 4, 5]
    path_fun = lambda format_str: [format_str.format(index) for index in data_index]

    for image_folder, detection_file_name, saved_file_name \
        in zip(path_fun(dataset_image_folder_format),
               path_fun(det_file_name_format),
               path_fun(saved_file_name_format)):
        print('start processing ' + saved_file_name)
        if osp.exists(saved_file_name):
            print('{} have already done !'.format(saved_file_name))
            continue

        tracker = BiTracker(affinity_model=affinity_model)
        mot_data_reader = MOTDataReader(image_folder=image_folder,
                                        detection_file_name=detection_file_name,
                                        min_confidence=0.0)

        result = np.array([])
        total_frame = 100
        # total_frame = len(mot_data_reader)
        for i, (img, det) in enumerate(mot_data_reader):  # loop for image
            if img is None or det is None or len(det) == 0:
                continue   # TODO how to update model for non-detection frame
            if detection_file_name.find("b4") != -1:
                det = det[det[:, 6] > 0.99, :]
            # if detection_file_name.find("b2") != -1:
            #     det = det[det[:, 6] > 0.99, :]

            heigth, width, _ = img.shape
            bbox = det[:, 2:6].astype(np.int32)  # bbox

            threshold = (9, 12)  # (width_th, height_th) (18,20) (12, 15) (9, 12)
            bbox = bbox[((bbox[:, 2] > threshold[0]) & (bbox[:, 3] > threshold[1]))]
            if len(bbox) == 0:
                continue

            bbox[:, 2:4] += bbox[:, :2]  # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            timer.tic()
            _ = tracker.update_tracker(i, img, bbox, force_init=False, total_frame=total_frame)
            timer.toc()
            print('{}:{}, {}/%\r'.format(os.path.basename(saved_file_name), i, int(i * 100 / len(mot_data_reader))))

            if i == total_frame - 1:
                break

        for id, t in enumerate(tracker.get_all_tracks()):
            res = []
            # id = t.id
            for n in t.get_tracklet_nodes():
                b = n.bbox
                if b is None:
                    continue
                res.append([n.frame_index + 1] + [id + 1] + [b[0], b[1], b[2] - b[0], b[3] - b[1]] + [-1, -1, -1, -1])
            res = np.array(res).astype(int)
            if len(result) == 0:
                result = res
            else:
                result = np.concatenate((result, res), axis=0)
        result = nms_interpolate(result)
        saved_file_dir = osp.dirname(saved_file_name)
        if not osp.isdir(saved_file_dir):
            mkdir_p(saved_file_dir)
        np.savetxt(saved_file_name, result[:, 0:6], fmt='%i', delimiter=',')

    print('The total running time is: {:.2f}. \n'.format(timer.total_time))
    print('The average running time is: {:.2f}. \n'.format(timer.average_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="H4tracker")
    args = parser.parse_args()
    test(args)
