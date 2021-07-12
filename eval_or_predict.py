# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision.transforms import InterpolationMode

import utils
import vision_transformer as vits

def eval_or_predict(model, args, eval=True):
    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val" if eval else "test"),
                                       transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu * 4,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_val)} val/test imgs.")

    # ============ building network ... ============
    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} is in use.")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    all_preds = []
    with torch.no_grad():
        for inp, target in metric_logger.log_every(val_loader, 50, header) if eval else val_loader:
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            _, output_cls = model(inp)
            _, preds = torch.max(output_cls, dim=1)
            all_preds.append(preds)

            num_labels = output_cls.shape[-1]
            if eval:
                acc1 = utils.accuracy(output_cls, target)[0]
                if num_labels >= 5:
                    acc5 = utils.accuracy(output_cls, target, topk=(5,))[0]
                loss = nn.CrossEntropyLoss()(output_cls, target)

            batch_size = inp.shape[0]
            if eval:
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                if num_labels >= 5:
                    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if eval:
            if num_labels >= 5:
                print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
            else:
                print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, losses=metric_logger.loss))

        predictions = torch.cat(all_preds)
        model.train()
        return predictions, {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    # TODO maybe make it a standalone
