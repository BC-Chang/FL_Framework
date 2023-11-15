from utils import get_dataloader
from torch.utils.data import DataLoader, random_split
import torch
from typing import List

def split_datasets(dataloader, NUM_CLIENTS):

    partition_size = len(dataloader) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(dataloader, lengths, torch.Generator().manual_seed(42))
    loaders = []
    for ds in datasets:
        loaders.append(DataLoader(ds, batch_size=1, shuffle=True))
    return loaders


def load_datasets_random_split(net_dict, BATCH_SIZE=1, NUM_CLIENTS=2, random_split=True):
    train_set = get_dataloader(net_dict, ['train'])['train']
    val_set = get_dataloader(net_dict, ['val'])['val']
    # test_set = get_dataloader(net_dict, ['test'])['test']
    #

    train_loader = split_datasets(train_set, NUM_CLIENTS)
    val_loader = split_datasets(val_set, NUM_CLIENTS)


    return train_loader, val_loader

def load_datasets_manual_split(data_dicts: List, BATCH_SIZE=1, NUM_CLIENTS=2):

    # Make sure that the number of data_dicts is equal to the number of clients
    assert len(data_dicts) == NUM_CLIENTS, "Number of data_dicts does not equal to the number of clients"

    train_loaders = []
    val_loaders = []
    for i in range(NUM_CLIENTS):
        train_data = get_dataloader(data_dicts[i], ['train'])['train']
        val_data = get_dataloader(data_dicts[i], ['val'])['val']
        # test_data = get_dataloader(data_dicts[i], ['test'])['test']


        train_loaders.append(DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True))
        val_loaders.append(DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False))

    return train_loaders, val_loaders