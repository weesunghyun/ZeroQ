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

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import os
try:
    from medmnist import INFO
    import medmnist
except ImportError:  # medmnist is optional
    INFO = None

# Number of classes for each MedMNIST subset
CLASSIFICATION_DATASETS = {
    'pathmnist': 9,
    'dermamnist': 7,
    'octmnist': 4,
    'pneumoniamnist': 2,
    'retinamnist': 5,
    'breastmnist': 2,
    'bloodmnist': 8,
    'tissuemnist': 8,
    'organamnist': 11,
    'organcmnist': 11,
    'organsmnist': 11,
}


class UniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = (torch.randint(high=255, size=self.size).float() -
                  127.5) / 5418.75
        return sample


def getRandomData(dataset='cifar10', batch_size=512, for_inception=False, init_data_path=None):
    """
    get random sample dataloader
    dataset: name of the dataset
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    dataset: name of MedMNIST subset if using MedMNIST
    init_data_path: the path to the initialization dataset
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 10000
    elif dataset == 'imagenet':
        num_data = 10000
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    elif dataset in CLASSIFICATION_DATASETS and INFO is not None:
        info = INFO[dataset]
        # MedMNIST datasets are standardized to 28x28 images
        img_size = 224  # All MedMNIST datasets use 28x28 images
        # size = (info['n_channels'], img_size, img_size)
        size = (3, img_size, img_size)
    else:
        raise NotImplementedError

    # Prepare dataset for initialization if provided
    dataset_obj = None
    if init_data_path is not None:
        init_resize_size = 256 if dataset == 'imagenet' or dataset in CLASSIFICATION_DATASETS else 32
        init_crop_size = 224 if dataset == 'imagenet' or dataset in CLASSIFICATION_DATASETS else 32

        init_transform = transforms.Compose([
            transforms.Resize(init_resize_size),
            transforms.CenterCrop(init_crop_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        dataset_obj = datasets.ImageFolder(init_data_path, transform=init_transform)
        length = len(dataset_obj)
        print(f"Init dataset loaded from {init_data_path}, {length} images")    
    else:
        dataset_obj = UniformDataset(length=10000, size=size, transform=None)

    data_loader = DataLoader(dataset_obj,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=16)
    return data_loader


def getTestData(dataset='imagenet',
                batch_size=1024,
                path='data/imagenet',
                for_inception=False):
    """
    Get dataloader of testset
    dataset: name of the dataset
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    dataset: name of MedMNIST subset if using MedMNIST
    """
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_dataset = datasets.ImageFolder(
            path + 'val',
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=16)
        return test_loader
    elif dataset == 'cifar10':
        data_dir = '/rscratch/yaohuic/data/'
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        test_dataset = datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        transform=transform_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=16)
        return test_loader
    elif dataset in CLASSIFICATION_DATASETS and INFO is not None:
        info = INFO[dataset]
        data_class = getattr(medmnist, info['python_class'])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
            transforms.Normalize(mean=[0.5] * info['n_channels'],
                                 std=[0.5] * info['n_channels'])
        ])
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Use size=28 for MedMNIST datasets (standard size)
        test_dataset = data_class(split='test', root=path,
                                 download=True, transform=transform, size=224)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=16)
        return test_loader
    else:
        raise NotImplementedError
