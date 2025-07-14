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
import random
from collections import OrderedDict
from datetime import datetime
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *

MEDMNIST_CLASSES = CLASSIFICATION_DATASETS


def convert_state_dict(pretrained_state_dict, new_model):
    """
    Converts a pretrained ResNet-18 state_dict to match the key format of the new model,
    including all 'num_batches_tracked' keys for BatchNorm layers.

    Args:
        pretrained_state_dict (OrderedDict): The state_dict object from the pretrained model.
        new_model (torch.nn.Module): An instance of the new model architecture.

    Returns:
        OrderedDict: The converted state_dict.
    """
    # For debugging: Check if the problematic key exists in the source state_dict
    if 'bn1.num_batches_tracked' not in pretrained_state_dict:
        print("Warning: 'bn1.num_batches_tracked' not found in the source checkpoint!")

    new_state_dict = OrderedDict()
    key_map = {}

    # 1. Initial block (conv1 and bn1)
    key_map.update({
        'conv1.weight': 'features.init_block.conv.conv.weight',
        'bn1.weight': 'features.init_block.conv.bn.weight',
        'bn1.bias': 'features.init_block.conv.bn.bias',
        'bn1.running_mean': 'features.init_block.conv.bn.running_mean',
        'bn1.running_var': 'features.init_block.conv.bn.running_var',
        'bn1.num_batches_tracked': 'features.init_block.conv.bn.num_batches_tracked'  # The missing key
    })

    # 2. ResNet stages (layer1 to layer4)
    for i in range(1, 5):  # Stages 1-4
        for j in range(2):  # Units 1-2 (for ResNet-18)
            # Body convolutions and their batchnorms
            for conv_idx in [1, 2]:
                old_prefix = f'layer{i}.{j}.conv{conv_idx}'
                new_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_prefix}.weight'] = f'{new_prefix}.conv.weight'

                old_bn_prefix = f'layer{i}.{j}.bn{conv_idx}'
                new_bn_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_bn_prefix}.weight'] = f'{new_bn_prefix}.bn.weight'
                key_map[f'{old_bn_prefix}.bias'] = f'{new_bn_prefix}.bn.bias'
                key_map[f'{old_bn_prefix}.running_mean'] = f'{new_bn_prefix}.bn.running_mean'
                key_map[f'{old_bn_prefix}.running_var'] = f'{new_bn_prefix}.bn.running_var'
                key_map[f'{old_bn_prefix}.num_batches_tracked'] = f'{new_bn_prefix}.bn.num_batches_tracked'

            # Downsample (identity) convolution for stages 2, 3, 4
            if i > 1 and j == 0:
                old_ds_prefix = f'layer{i}.{j}.downsample'
                new_ds_prefix = f'features.stage{i}.unit{j+1}.identity_conv'
                key_map[f'{old_ds_prefix}.0.weight'] = f'{new_ds_prefix}.conv.weight'
                key_map[f'{old_ds_prefix}.1.weight'] = f'{new_ds_prefix}.bn.weight'
                key_map[f'{old_ds_prefix}.1.bias'] = f'{new_ds_prefix}.bn.bias'
                key_map[f'{old_ds_prefix}.1.running_mean'] = f'{new_ds_prefix}.bn.running_mean'
                key_map[f'{old_ds_prefix}.1.running_var'] = f'{new_ds_prefix}.bn.running_var'
                key_map[f'{old_ds_prefix}.1.num_batches_tracked'] = f'{new_ds_prefix}.bn.num_batches_tracked'

    # 3. Final fully-connected layer
    key_map.update({
        'fc.weight': 'output.weight',
        'fc.bias': 'output.bias'
    })

    # Populate the new_state_dict using the generated map
    for old_key, new_key in key_map.items():
        if old_key in pretrained_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[old_key]

    return new_state_dict


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
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='random seed for reproducibility')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='path to dataset')
    parser.add_argument('--init_data_path',
                        type=str,
                        default=None,
                        help='path to initialization dataset')
    args = parser.parse_args()
    return args


def create_logger(args):
    log_dir = os.path.join(
        'log',
        f"{args.dataset}_{args.model}_w{args.weight_bit}a{args.act_bit}{'_' + os.path.basename(args.init_data_path) if args.init_data_path is not None else ''}",
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
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    logger = create_logger(args)
    # Enable deterministic behavior when a seed is provided
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load pretrained model
    if args.dataset in MEDMNIST_CLASSES:
        num_classes = MEDMNIST_CLASSES.get(args.dataset, 2)
        model = ptcv_get_model(
            args.model, pretrained=False, num_classes=num_classes)
        if args.pretrained is not None:
            checkpoint = torch.load(args.pretrained)
            if isinstance(checkpoint, dict) and 'net' in checkpoint:
                converted_state_dict = convert_state_dict(checkpoint['net'], model)
            else:
                converted_state_dict = convert_state_dict(checkpoint, model)
            model.load_state_dict(converted_state_dict)
            logger.info('****** Converted and loaded pretrained weights ******')
    else:
        if args.pretrained is not None:
            model = ptcv_get_model(args.model, pretrained=False)
            model.load_state_dict(torch.load(args.pretrained))
        else:
            model = ptcv_get_model(args.model, pretrained=True)
    logger.info('****** Full precision model loaded ******')

    # Load validation data
    default_path = './data/medmnist/' if args.dataset in MEDMNIST_CLASSES else './data/imagenet/'
    data_path = args.data_path if args.data_path is not None else default_path
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path=data_path,
                              for_inception=args.model.startswith('inception'))
    # Generate distilled data
    dataloader = getDistilData(
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'),
        init_data_path=args.init_data_path
        )
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
