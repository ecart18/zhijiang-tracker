# zhijiang-tracker, a tracking by detection algorithm for ZhiJiang multi pedestrian tracking challenge

By [Tao Hu](https://ecart18.github.io/).

The code for the official implementation of . A PyTorch implementation of zhijiang MOT chanllenge, **zhijiang-tracker**, with support for training and test.

The code will be made publicly available shortly.

## model description
#### our methods consist of three parts including detection model (YOLOv3), person reid model (triplet loss and hard sampling mining), and tracking model (H4tracker)


## requirements preparing
    $ pip install -r requirements.txt      


## pretrain model download

#### reid model
download pretrain model resnet50 "resnet50-19c8e357.pth" and put it in ./pretrainmodel/

#### detection model
download pretain detection model
    $ cd pretrainmodel/yolo3/data/
    $ bash download_weights.sh


## data download

#### download and prepare coco dataset for person detection
    $ cd data/yolov3/coco/
    $ bash get_coco_dataset.sh

##### down load reid data
download reid data from [website](https://docs.google.com/forms/d/e/1FAIpQLSfueNRWgRp3Hui2HdnqHGbpdLUgSn-W8QxpZF0flcjNnvLZ1w/viewform?formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0) and put it in ./data/reid/cuhk03/raw

##### prepare test datasets: zhijiang MOT challenge level2, put origin video b1...b5 in ./data/zj_test/level2_video/
    $ cd ./data/zj_test
    $ python video2frames.py


## training
#### the reid and detection model are saved in model/reid and model/yolo3/ respectively
cd experiments/
    $ bash run_training.sh

## test
#### the results of detection and tracking are saved in result/level2/det and result/level2/trk respectively
cd experiments/
    $ bash run_test.sh

## Thanks to some open projects
- [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [open-reid](https://github.com/leonardbereska/openreid)

## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
