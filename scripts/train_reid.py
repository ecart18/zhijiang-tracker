from __future__ import print_function, absolute_import

import argparse
import sys
import numpy as np
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from ReID.model.backbone import get_backbone
from ReID.model.head import get_head
from ReID.model.loss import build_loss
from ReID.trainer.trainer import Trainer
from ReID.evaluator.evaluator import Evaluator
from ReID.datasets.dataloader import build_dataloader
from ReID.utils.logger import Logger
from ReID.utils.osutils import load_checkpoint, save_checkpoint


def main(args):

    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join('./logs', 'reid_train_log.txt'))

    # Create data loaders
    dataset, num_classes, train_loader, val_loader, test_loader = build_dataloader()

    backbone = get_backbone()
    model = get_head(backbone)

    # Load from checkpoint
    resume = ""
    start_epoch = best_top1 = 0
    if resume:
        print("load checkpoint file from {} \n".format(resume))
        checkpoint = load_checkpoint(resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)

    # Criterion
    criterion = build_loss().cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002,
                                 weight_decay=5e-4)
    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = 0.0002 if epoch <= 100 else \
            0.0002 * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, 150):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join("../model/reid", 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID")
    args = parser.parse_args()
    main(parser.parse_args())
