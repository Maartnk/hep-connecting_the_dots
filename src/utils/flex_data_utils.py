import torch
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging

import sys
import numpy as np

sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray

PAD_TOKEN = -1

def load_dataloader(config, device, mode="all"):
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["dataloader_num_workers"]

    loaders = {}

    if mode == "train" or mode == "all":
        shuffle = config["training"]["shuffle"]
        logging.info("Loading train data with DataLoader")
        train_dataset = ForwardPassDataset(data_dir, config["data"]["train_file"])
        val_dataset = ForwardPassDataset(data_dir, config["data"]["val_file"])
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if mode == "eval" or mode == "all":
        test_dataset = ForwardPassDataset(data_dir, config["data"]["test_file"])
        test_helper_dataset = ScoringHelperDataset(
            data_dir, config["data"]["test_helperfile"]
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["test_helper"] = DataLoader(
            test_helper_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return loaders


def load_truths(config):
    data_dir = config["data"]["data_dir"]
    test_truth_filename = config["data"]["test_truthfile"]
    test_truth_filepath = os.path.join(data_dir, test_truth_filename)
    return pd.read_csv(test_truth_filepath)

def round_up_to_multiple(x, multiple = 128):
    return ((x + multiple - 1) // multiple) * multiple

def collate_fn(batch):
    batch_dim = len(batch[0])
    
    if batch_dim == 3:
        dat1, dat2, dat3 = zip(*batch)
        seq_lengths = torch.tensor([x.size(0) for x in dat1], dtype=torch.long)
        max_seq_len = max(seq_lengths).item()
        padded_len = round_up_to_multiple(max_seq_len)

        def pad_to_len(tensors, target_len, pad_value=PAD_TOKEN):
            padded = []
            for t in tensors:
                pad_len = target_len - t.size(0)
                if pad_len == 0:
                    padded.append(t)
                else:
                    if t.dim() == 1:                     # labels
                        padded.append(F.pad(t, (0, pad_len), value=pad_value))
                    elif t.dim() == 2:                   # coords, params
                        padded.append(F.pad(t, (0, 0, 0, pad_len), value=pad_value)) # Pad along dimension 0 (sequence length)
                    else:
                        raise ValueError(f"Unexpected tensor rank {t.dim()} in collate_fn.")
            return torch.stack(padded)

        dat1_padded = pad_to_len(dat1, padded_len)
        dat2_padded = pad_to_len(dat2, padded_len)
        dat3_padded = pad_to_len(dat3, padded_len)

        return dat1_padded, dat2_padded, dat3_padded, seq_lengths

    elif batch_dim == 2:
        dat1, dat2 = zip(*batch)
        seq_lengths = torch.tensor([x.size(0) for x in dat1], dtype=torch.long)
        max_seq_len = max(seq_lengths).item()
        padded_len = round_up_to_multiple(max_seq_len)

        def pad_to_len(tensors, target_len, pad_value=PAD_TOKEN):
            padded = []
            for t in tensors:
                pad_len = target_len - t.size(0)
                if pad_len == 0:
                    padded.append(t)
                else:
                    if t.dim() == 1:                     # labels
                        padded.append(F.pad(t, (0, pad_len), value=pad_value))
                    elif t.dim() == 2:                   # coords, params
                        padded.append(F.pad(t, (0, 0, 0, pad_len), value=pad_value)) # Pad along dimension 0 (sequence length)
                    else:
                        raise ValueError(f"Unexpected tensor rank {t.dim()} in collate_fn.")
            return torch.stack(padded)

        dat1_padded = pad_to_len(dat1, padded_len)
        dat2_padded = pad_to_len(dat2, padded_len)

        return dat1_padded, dat2_padded, seq_lengths

    else:
        raise ValueError(
            f"Unexpected sample length {batch_dim}. "
            "Expected 2 or 3 elements per sample."
        )

def collate_fn_scoringhelper(batch):
    dat1, dat2 = zip(*batch)
    dat1_padded = pad_sequence(dat1, batch_first=True, padding_value=PAD_TOKEN)
    dat2_padded = pad_sequence(dat2, batch_first=True, padding_value=PAD_TOKEN)
    return dat1_padded, dat2_padded


class ForwardPassDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.coords, self.labels = torch.load(file_path, weights_only = False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        coords_tensor = torch.tensor(self.coords[idx][:, :3], dtype=torch.float)
        params_tensor = torch.tensor(self.coords[idx][:, 3:], dtype=torch.float)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return coords_tensor, params_tensor, labels_tensor
    
    def get_hits_xyz(self, idx):
        return torch.tensor(self.coords[idx][:, :3], dtype=torch.float)
        


class ScoringHelperDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.hit_ids, self.event_ids = torch.load(file_path, weights_only = False)

    def __len__(self):
        return len(self.hit_ids)

    def __getitem__(self, idx):
        hit_ids_tensor = torch.tensor(self.hit_ids[idx], dtype=torch.long)
        event_ids_tensor = torch.tensor(self.event_ids[idx], dtype=torch.long)
        return hit_ids_tensor, event_ids_tensor
