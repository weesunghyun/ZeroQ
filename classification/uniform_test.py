#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
import numpy as np
import torch.nn as nn
import logging
import os
from datetime import datetime
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *

MEDMNIST_CLASSES = CLASSIFICATION_DATASETS


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'] + list(MEDMNIST_CLASSES.keys()),
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='path to pretrained weights')
    parser.add_argument('--weight_bit',
                        type=int,
                        default=8,
                        help='bitwidth for weights')
    parser.add_argument('--act_bit',
                        type=int,
                        default=8,
                        help='bitwidth for activations')
    args = parser.parse_args()
    return args


def create_logger(args):
    log_dir = os.path.join(
        'log',
        f"{args.dataset}_{args.model}_{args.weight_bit}w_{args.act_bit}a",
    )
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{timestamp}.log')

    logger = logging.getLogger('zeroq')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


if __name__ == '__main__':
    args = arg_parse()
    logger = create_logger(args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    if args.dataset in MEDMNIST_CLASSES:
        num_classes = MEDMNIST_CLASSES.get(args.dataset, 2)
        model = ptcv_get_model(
            args.model, pretrained=False, num_classes=num_classes)
        if args.pretrained is not None:
            model.load_state_dict(torch.load(args.pretrained))
    else:
        if args.pretrained is not None:
            model = ptcv_get_model(args.model, pretrained=False)
            model.load_state_dict(torch.load(args.pretrained))
        else:
            model = ptcv_get_model(args.model, pretrained=True)
    logger.info('****** Full precision model loaded ******')

    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/medmnist/' if args.dataset in MEDMNIST_CLASSES else './data/imagenet/',
                              for_inception=args.model.startswith('inception'))
    # Generate distilled data
    dataloader = getDistilData(
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'))
    logger.info('****** Data loaded ******')

    # Quantize single-precision model
    quantized_model = quantize_model(
        model, weight_bit=args.weight_bit, act_bit=args.act_bit)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # Update activation range according to distilled data
    update(quantized_model, dataloader)
    logger.info('****** Zero Shot Quantization Finished ******')

    # Freeze activation range during test
    freeze_model(quantized_model)
    quantized_model = nn.DataParallel(quantized_model).cuda()

    # Test the final quantized model
    test(quantized_model, test_loader, logger)
