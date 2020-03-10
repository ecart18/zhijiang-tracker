from __future__ import absolute_import

import os
import cv2
import pandas as pd


class MOTDataReader:
    def __init__(self, image_folder, detection_file_name, min_confidence=0.0):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, '{0:06d}.jpg')
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        self.detection = self.detection[self.detection[6] >= min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

    def __len__(self):
        return max(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if index > max(self.detection_group_keys) or self.detection_group_keys.count(index) == 0:
            return None
        if self.detection_group.get_group(index).values.shape[1] == 10 \
          or self.detection_group.get_group(index).values.shape[1] == 7:  # TODO modify
            return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > max(self.detection_group_keys):
            return None
        return cv2.cvtColor(cv2.imread(self.image_format.format(index)), cv2.COLOR_BGR2RGB)

    def __getitem__(self, item):
        return (self.get_image_by_index(item+1),
                self.get_detection_by_index(item+1))
