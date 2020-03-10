
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
from terminaltables import AsciiTable

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from YOLOV3.model.models import Darknet
from YOLOV3.utils.logger import TFLogger, Logger
from YOLOV3.utils.utils import load_classes, weights_init_normal
from YOLOV3.utils.datasets import ListDataset
from YOLOV3.evaluate.evaluate import evaluate


def main(args):
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.stdout = Logger(osp.join('./logs', 'yolov3_train_log.txt'))
    logger = TFLogger("logs")

    # COCO data
    train_path = "../data/yolo3/coco/coco/trainvalno5k.txt"
    valid_path = "../data/yolo3/coco/coco/5k.txt"
    class_names = load_classes("../data/yolo3/coco/coco/coco.names")

    # Initiate model
    model = Darknet("../YOLOV3/config/yolov3.cfg", img_size=416).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    pretrained_weights = "../pretrainmodel/yolo3/yolov3.weights"
    if pretrained_weights:
        if pretrained_weights.endswith(".pth"):
            print("load pre-trained weights in {}".format(pretrained_weights))
            model.load_state_dict(torch.load(pretrained_weights))
        else:
            print("load pre-trained weights in {}".format(pretrained_weights))
            model.load_darknet_weights(pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True,
                          multiscale=True,
                          img_size=416)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             collate_fn=dataset.collate_fn,
                                             batch_size=8,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(150):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % 2:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, 50, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if valid_path:
            eval_interval = 1
            if epoch % eval_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=416,
                    batch_size=8,
                )
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")

        save_model_interval = 1
        if epoch % save_model_interval == 0:
            torch.save(model.state_dict(), osp.join("../model/yolo3/", 'yolov3_ckpt_{:0>3d}.pth'.format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOV3_TRAIN")
    args = parser.parse_args()
    main(parser.parse_args())










