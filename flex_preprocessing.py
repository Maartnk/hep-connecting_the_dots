import torch
import numpy as np
import pandas as pd
import random
import math

from torch.utils.data import TensorDataset, random_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import psutil, os, gc
def _rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9
def mem(msg):
    gc.collect()
    print(f"[MEM] {msg}: {_rss_gb():.2f} GB")

PAD_TOKEN = 0

data_path = "[DATA_PATH]"
normalize = False
chunking = False

usecols = ["x","y","z","px","py","pz","q","weight","event_id","particle_id"]

if not chunking:
    sampled_data = pd.read_csv(data_path, header=0, sep=',', usecols=usecols)
    mem("after read_csv")

# Normalize the data if applicable
if normalize:
    for col in ["x", "y", "z", "px", "py", "pz", "q"]:
        mean = sampled_data[col].mean()
        std = sampled_data[col].std()
        sampled_data.loc[:, col] = (sampled_data[col] - mean)/std

# Shuffling the data and grouping by event ID
#shuffled_data = sampled_data.sample(frac=1, random_state=13) # For if batch based padding is desired

#data_grouped_by_event = shuffled_data.groupby("event_id", group_keys=False) # For if event based padding is desired (PADDING LOGIC NOT IMPLEMENTED)

def extract_track_params_data(data):
    data['p'] = np.sqrt(data['px']**2 + data['py']**2 + data['pz']**2)
    data['theta'] = np.arccos(data['pz']/data['p'])
    data['phi'] = np.arctan2(data['py'], data['px'])
    data['sin_phi'] = np.sin(data['phi'])
    data['cos_phi'] = np.cos(data['phi'])
    return data

data_track_params = extract_track_params_data(sampled_data) # For if batch based padding is desired

data = data_track_params[['x', 'y', 'z', 'event_id', 'particle_id', 'q', 'theta', 'phi', 'sin_phi', 'cos_phi', 'p', 'weight']]
mem("after data_track_params")

del sampled_data, data_track_params
gc.collect()

#CALC CLASS_ID
n_bins_phi = 30
n_bins_theta = 30
n_bins_p = 3
n_bins_q = 2

data['p_bin'] = pd.qcut(data['p'], q=n_bins_p, labels=False)
data['theta_bin'] = pd.qcut(data['theta'], q=n_bins_theta, labels=False)
data['phi_bin'] = pd.qcut(data['phi'], q=n_bins_phi, labels=False)
data['q_bin'] = data['q'].apply(lambda x: 0 if x == -1 else 1)

# ADDING VALUES BY 1 SO THAT I CAN USE 0 FOR PADDING!
data['class_id'] = 1 + data['q_bin'] * n_bins_phi * n_bins_theta * n_bins_p \
                    + data['phi_bin'] * n_bins_theta * n_bins_p \
                    + data['theta_bin'] * n_bins_p \
                    + data['p_bin']

mem("after binning/class_id")

data['hit_id'] = data.index

n_unique_classes = len(np.unique(data['class_id']))

data['unique_tracks_in_group'] = data.groupby(['event_id', 'class_id'])['particle_id'].transform('nunique')
problematic_mask = data['unique_tracks_in_group'] > 1
problematic_rows = data[problematic_mask]
unique_problematic_tracks = problematic_rows[['event_id', 'particle_id']].drop_duplicates()
wrong_tracks_count = len(unique_problematic_tracks)

total_tracks_count = data[['event_id', 'particle_id']].drop_duplicates().shape[0]
wrong_tracks_percentage = (wrong_tracks_count / total_tracks_count)

mem("after complete class_id")


#SUBSET CREATION
save_dir = "[SAVE_DIRECTORY_PATH]"
os.makedirs(save_dir, exist_ok=True)

train_frac, val_frac = 0.6, 0.2 #(test_frac is implicitly 1 - train_frac - val_frac)

unique_event_id = data['event_id'].unique()
n_events = len(unique_event_id)
shuffled_events = np.random.default_rng(seed = 42).permutation(unique_event_id)

n_train = int(train_frac * n_events)
n_val = int(val_frac * n_events)

train_event_ids = shuffled_events[:n_train]
val_event_ids = shuffled_events[n_train:n_train + n_val]
test_event_ids = shuffled_events[n_train + n_val:]

train_set = data[data['event_id'].isin(train_event_ids)].reset_index(drop=True)
val_set = data[data['event_id'].isin(val_event_ids)].reset_index(drop=True)
test_set = data[data['event_id'].isin(test_event_ids)].reset_index(drop=True)

print(f"Train: {train_set.shape[0]} rows ({len(train_event_ids)} events)")
print(f"Val: {val_set.shape[0]} rows ({len(val_event_ids)} events)")
print(f"Test: {test_set.shape[0]} rows ({len(test_event_ids)} events)")

test_set[['hit_id', 'particle_id', 'event_id', 'weight']].to_csv(f"{save_dir}/test_truths_7000.csv", index=False)
mem("after split")

def extract_and_shuffle_sequences(df, feature_cols, label_col, seed=None):
    """Extract per-event sequences of features and labels, and shuffle hits within each event."""
    rng = np.random.default_rng(seed) if seed is not None else None
    hits_list, classes_list = [], []
    for eid, grp in df.groupby('event_id', sort=False):
        feats = grp[feature_cols].to_numpy(copy=False)
        labs  = grp[label_col].to_numpy(copy=False)
        idx = np.arange(labs.shape[0], dtype=np.int32)
        if rng is not None:
            rng.shuffle(idx)
        else:
            np.random.shuffle(idx)
        hits_list.append(feats[idx])
        classes_list.append(labs[idx])
    return hits_list, classes_list

def sort_sequences_by_length(features, labels, descending = False):
    """Sort sequences by length (number of hits) to improve batching efficiency."""
    lengths = np.fromiter((len(seq) for seq in labels), dtype=np.int32)
    order   = np.argsort(lengths)
    if descending:
        order = order[::-1]

    sorted_feats   = [features[i] for i in order]
    sorted_labels  = [labels[i]  for i in order]
    
    return sorted_feats, sorted_labels

def save_subset(features, label, name):
    """Save a dataset split to disk."""
    torch.save((features, label), f"{save_dir}/data_{name}.pt")

    n_events   = len(features)
    seq_lengths = [len(seq) for seq in features]
    n_rows     = sum(seq_lengths)
    min_len    = min(seq_lengths) if seq_lengths else 0
    max_len    = max(seq_lengths) if seq_lengths else 0

    print(
        f"{name:5} → {n_rows:7d} rows, across {n_events:5d} events\n"
        f"event-lengths = [{min_len} … {max_len}]"
    )

feature_cols = ['x','y','z','theta','sin_phi', 'cos_phi','q']
label_col    = 'class_id'

train_hits_shuffled_eid, train_classes_shuffled_eid = extract_and_shuffle_sequences(train_set, feature_cols, label_col, seed=13)
val_hits_shuffled_eid, val_classes_shuffled_eid = extract_and_shuffle_sequences(val_set, feature_cols, label_col, seed=13)
test_hits_shuffled_eid, test_classes_shuffled_eid = extract_and_shuffle_sequences(test_set, feature_cols, label_col, seed=13)

train_hits, train_classes = sort_sequences_by_length(train_hits_shuffled_eid, train_classes_shuffled_eid)
val_hits, val_classes = sort_sequences_by_length(val_hits_shuffled_eid, val_classes_shuffled_eid)
test_hits, test_classes = sort_sequences_by_length(test_hits_shuffled_eid, test_classes_shuffled_eid)

mem("after building sequences")

save_subset(train_hits, train_classes, "sorted_train")
save_subset(val_hits, val_classes, "sorted_val")
save_subset(test_hits, test_classes,  "sorted_test")

### SAVE TEST HELPER
grouped_test = test_set.groupby('event_id', sort=False)
rng = np.random.default_rng(seed=13)

test_hits = []
test_classes = []
test_hit_ids = []
test_event_ids = []
for event_id, group in grouped_test:
    feature_coords = group[feature_cols].values
    class_ids = group[label_col].values
    hit_ids = group['hit_id'].values
    event_ids = group['event_id'].values
    
    indices = np.arange(len(feature_coords))
    #np.random.shuffle(indices)
    rng.shuffle(indices)
    class_ids = class_ids[indices]
    hit_ids = hit_ids[indices]
    event_ids = event_ids[indices]
    
    test_classes.append(class_ids)
    test_hit_ids.append(hit_ids)
    test_event_ids.append(event_ids)

sorted_test_indices = np.argsort([len(seq) for seq in test_classes])
sorted_test_hit_ids = [test_hit_ids[i] for i in sorted_test_indices]
sorted_test_event_ids = [test_event_ids[i] for i in sorted_test_indices]

torch.save((sorted_test_hit_ids, sorted_test_event_ids), f"{save_dir}/sorted_test_helper.pt")