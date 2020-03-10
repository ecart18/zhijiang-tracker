
import cv2
import numpy as np
import copy

from .bbox import overlap_ratio


def img_norm_resnet(img):
    def img_normalize(img, mean, std):
        img = (1 + img) / 256
        img = (np.transpose(img, (2, 0, 1)).astype(np.float32) - mean) / std
        return img.astype(np.float32)
    img = img.astype(np.float32)
    mean = np.expand_dims(np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=1), axis=1).astype(np.float32)
    std = np.expand_dims(np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=1), axis=1).astype(np.float32)
    img = img_normalize(img, mean, std)
    return img


class InferenceUtil:

    @staticmethod
    def convert_image2patch(image, bbox):
        """
        crop and transform image to the FloatTensor (1, 3, size, size)
        :param image: original image
        :param bbox: bbox [x1, y1, x2, y2]
        :return: the transformed image FloatTensor (i.e. 1 x 3 x height x width)
        """
        patchs_bbox = copy.copy(bbox).astype(np.int32)
        patchs_bbox[patchs_bbox < 0.] = 0.
        det_num = patchs_bbox.shape[0]
        det_patchs = np.zeros((det_num, 3, 256, 128)).astype(np.float32)
        for i in range(det_num):
            bb = patchs_bbox[i, :]
            image_cropped = image[bb[1]:(bb[3] + 1), bb[0]:(bb[2] + 1), :]
            image_cropped = cv2.resize(image_cropped, (128, 256))
            image_cropped = img_norm_resnet(image_cropped)
            det_patchs[i, :] = image_cropped

        img_size = np.array(image.shape[0:2]).astype(np.float32)
        img_size = np.repeat(np.expand_dims(img_size, axis=0), axis=0, repeats=det_num)
        return det_patchs, bbox, img_size

    @staticmethod
    def get_iou(pre_boxes, next_boxes):

        h = len(next_boxes)
        w = len(pre_boxes)
        if h == 0 or w == 0:
            return []
        iou = np.zeros((h, w), dtype=float)
        for i in range(h):
            rect1 = np.expand_dims(next_boxes[i, :], 0)
            rect1 = np.repeat(rect1, pre_boxes.shape[0], axis=0)
            iou[i,:] = overlap_ratio(rect1, pre_boxes)
        return iou





