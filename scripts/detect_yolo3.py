
import os
import sys
import os.path as osp
import time
import datetime
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from YOLOV3.model.models import Darknet
from YOLOV3.utils.utils import load_classes, non_max_suppression, rescale_boxes
from YOLOV3.utils.datasets import ImageFolder
from YOLOV3.utils.detection_filter_by_bus import filter_person


def main(args):
    cudnn.benchmark = True
    det_img_size = 1248
    det_cls = ['person', 'bus']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet("../YOLOV3/config/yolov3.cfg", img_size=det_img_size).to(device)
    
    det_weight_path = '../model/yolo3/yolov3_ckpt_001.pth'
    if det_weight_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(det_weight_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(det_weight_path))

    model.eval()  # Set in evaluation mode

    det_image_folder = "../data/zj_test/level2"
    det_det_result_path = "../result/level2/det"
    all_videos = sorted([os.path.join(det_image_folder, i) for i in os.listdir(det_image_folder)])
    for image_folder in all_videos:

        if os.path.isdir(os.path.join(det_det_result_path, image_folder.split('/')[-1])):
            print("{} is done".format(os.path.join(det_det_result_path, image_folder.split('/')[-1])))
            continue

        dataloader = DataLoader(
            ImageFolder(os.path.join(image_folder, 'img1'), img_size=det_img_size),
            batch_size=4,
            shuffle=False,
            num_workers=8)

        classes = load_classes("../data/yolo3/coco/coco/coco.names")  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, conf_thres=0.5, nms_thres=0.5)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        result_file_name = os.path.join(det_det_result_path, image_folder.split('/')[-1])
        if not os.path.isdir(result_file_name):
            os.makedirs(result_file_name)
        result_file_name = os.path.join(result_file_name, 'det.txt')
        result = np.array([])
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            img = np.array(Image.open(path))
            if detections is not None:
                res = []
                # Rescale boxes to original image
                detections = rescale_boxes(detections, det_img_size, img.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if classes[int(cls_pred)] in det_cls:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        res.append([img_i + 1] + [-1] + [x1, y1, box_w, box_h] + [cls_conf, int(cls_pred), conf, -1])
            res = np.array(res)
            res = filter_person(res)

            if len(result) == 0:
                result = res
            else:
                if len(res) != 0:
                    result = np.concatenate((result, res), axis=0)

        np.savetxt(result_file_name,
                   result,
                   fmt=','.join(['%d']*2) + ',' + ','.join(['%0.3f']*5) + ',%d,%0.3f,%d',
                   delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOV3_DET")
    args = parser.parse_args()
    main(parser.parse_args())






