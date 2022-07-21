import os
import math
import logging
import random
import argparse
from time import time
import glob
import sys
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import timm
from timm.utils import accuracy
from torch.utils.data import dataloader
from utils import misc
from utils import sampler
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from sklearn.metrics import f1_score, accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    val_metric = AccF1Metric(ignore_index=None)
    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            model = model.cuda()
            output = model(images)
            loss = criterion(output, target)

        output = torch.nn.functional.softmax(output, dim=-1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        val_metric.update(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    val_acc, val_f1 = val_metric.get()
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss{losses.global_avg:.3f}'.
          format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print("Val Acc: {:>3.3f}%  F1: {:>3.3f}% ".format(val_acc, val_f1))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def acc_f1_score(y_true, y_pred, ignore_index=None, normalize=False, average='macro', **kwargs):
    """Multi-class f1 score and accuracy"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if ignore_index is not None:
        leave = y_true != ignore_index
        y_true = y_true[leave]
        y_pred = y_pred[leave]
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average, **kwargs)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=normalize)
    return acc, f1
class AccF1Metric(object):
    def __init__(self, ignore_index, average='macro'):
        self.ignore_index = ignore_index
        self.average = average
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().reshape(-1)
        self.y_pred.append(y_pred)
        self.y_true.append(y_true.detach().cpu().numpy())

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def get(self):
        # y_true = np.stack(self.y_true, axis=0).reshape(-1)
        # y_pred = np.stack(self.y_pred, axis=0).reshape(-1)
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        acc, f1 = acc_f1_score(y_true=y_true, y_pred=y_pred,
                               average=self.average,
                               normalize=True,
                               ignore_index=self.ignore_index)
        return acc, f1
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    print_freq = 2
    accum_iter = args.accum_iter
    train_metric = AccF1Metric(ignore_index=None)
    if log_writer is not None:
        print('log_dir:{}'.format(log_writer.log_dir))
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(samples)
        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss = criterion(outputs, targets)


        train_metric.update(outputs, targets)
        train_acc, train_f1 = train_metric.get()
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_value = loss.item()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if not math.isfinite(loss_value):
            print("loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', warmup_lr, epoch_1000x)
            print("Train Acc: {:>3.3f}%  F1: {:>3.3f}% ".format(train_acc, train_f1))
            print(f"epoch: {epoch}, step:{data_iter_step}, loss:{loss},lr:{warmup_lr}")
    train_acc, train_f1 = train_metric.get()
    return train_acc, train_f1

def build_transform(is_train, args):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        print("train transform")
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                torchvision.transforms.ToTensor(),
            ]
        )
    print("eval transform")
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor(),

        ]
    )


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    print(f"finding classes from {path}: \t{info[0]}")
    print(f"mapping classes from {path} to indexes \t{info[1]}")
    return dataset


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=72, type=int,
                        help='')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    # model
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--lr', default=0.0001, metavar='LR')
    parser.add_argument('--root_path', default='F:/ABAW2022/codezip/LSDresnet18/data')
    parser.add_argument('--output_dir', default='./output_dir_pretrained')
    parser.add_argument('--log_dir', default='./output_dir_pretrained')
    parser.add_argument('--model', default='hrnet_w18')
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args, mode='train', test_image_path=''):
    print(f"{mode} mode...")
    if mode == 'train':
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        sampler_train = sampler.RandomSampler(dataset_train)
        sampler_val = sampler.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        model = timm.create_model(args.model, pretrained=True, num_classes=6, drop_rate=0.1)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params (M):%.2f' % (n_parameters / 1.e6))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

        for epoch in range(args.start_epoch, args.epochs):

            print(f"epoch {epoch}")
            print(f"length if data_loader_train is {len(data_loader_train)}")

            if epoch % 1 == 0:
                print("evaluating...")
                model.eval()
                test_stats = evaluate(data_loader_val, model, device)

                print(f"accuracy of the network on the {len(dataset_val)} test image :{test_stats['acc1']:.1f}%")
                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
                model.train()
            print("training...")
            train_acc, train_f1 = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch + 1,
                                          loss_scaler, None, log_writer=log_writer, args=args)
            print("Train Acc: {:>3.3f}%  F1: {:>3.3f}% ".format(train_acc, train_f1))
            if args.output_dir:
                print("saving checkpoints...")
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=epoch
                )
            # logging.info(
            #     f'Total Loss,{}, Ex:{ex_loss_record.avg}, AU:{au_loss_record.avg}, VA:{va_loss_record.avg}')
    # else:
    #     model = timm.create_model(args.model, pretrained=True, num_classes=6, drop_rate=0.1)
    #     class_dict = {
    #         'ANGRER': 0, 'DISGUST': 1, 'FEAR': 2, 'HAPPINESS': 3, 'SADNESS': 4, 'SURPRISE': 5
    #     }
    #     n_parameters = sum(p.numel() for p in model.parameters if p.requires_grad)
    #     print('number of trainable params (M):%.2f' % (n_parameters / 1.e6))
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     loss_scaler = NativeScaler()
    #     misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    #     model.eval()
    #     image = Image.open(test_image_path).convert('RGB')
    #     image = image.resize((args.input_size, args.input_size), Image.ANTIALIAS)
    #     image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    #     with torch.no_grad():
    #         output = model(image)
    #
    #     output = torch.nn.functional.softmax(output, dim=-1)
    #     class_idx = torch.argmax(output, dim=1)[0]
    #     score = torch.max(output, dim=1)[0][0]
    #     print(f"image path is {test_image_path}")
    #     print(
    #         f"score is {score.item()}, class is {class_idx.item()}, class name is {list(class_dict.keys())[list(class_dict.values()).index(class_idx)]}")
    #     time.sleep(0.5)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log_file_name = r'./log.txt'
    # logging.basicConfig(filename=log_file_name, level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger()
    # print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'train'
    if mode == 'train':
        main(args, mode=mode)
    # else:
    #     image = glob.glob('F:/ABAW2022/codezip/LSDresnet18/data/test/*/*.jpg')
    #     random.shuffle(image)
    #
    #     for image in image:
    #         print('\n')
    #         main(args, mode=mode, test_image_path=image)
