import math
import os
import random

import numpy as np
from data.select_dataset import define_Dataset
from utils import utils_option as option
import torch
from torch.utils.data import DataLoader
from utils.caching_dataloader import EpochCachingDataLoader

MAX_EPOCH = 1000000

opt = option.parse('options/train_fbcnn_color.json', is_train=True)

opt = option.dict_to_nonedict(opt)

seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
random.seed(seed)

dataset_type = opt['datasets']['train']['dataset_type']

dataset_opt = opt['datasets']['train']


train_set = define_Dataset(dataset_opt)
print('Dataset [{:s} - {:s}] is created.'.format(train_set.__class__.__name__, dataset_opt['name']), flush=True)
train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
train_loader = DataLoader(train_set,
                            batch_size=dataset_opt['dataloader_batch_size'],
                            shuffle=dataset_opt['dataloader_shuffle'],
                            num_workers=6,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True)
batch_size = dataset_opt['dataloader_batch_size']
cache_dir = os.path.join(opt['path']['root'], 'cached_batches', f'cached_batches_{batch_size}')

print(f"cache_dir: {cache_dir}", flush=True)
starting_epoch = 503

custom_train_loader = EpochCachingDataLoader(train_loader, cache_dir, current_epoch=starting_epoch)


for epoch in range(starting_epoch, MAX_EPOCH):
    for i, train_data in enumerate(custom_train_loader):
        pass