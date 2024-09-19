import torch
from torch.nn.utils.rnn import pad_sequence

from config import Config

config = Config()

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and labels.
    """
    # Extract features and labels from the batch
    features_list = [item['features'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # Pad features
    # Find the maximum sequence length in the batch
    max_seq_len = config.seq_len
    # Find the maximum number of frames per slice
    frames_per_slice = config.time_steps  # Assuming consistent frames_per_slice
    n_features = features_list[0].shape[1]        # Assuming consistent n_features

    # Pad sequences to the maximum length
    padded_features = []
    feature_masks = []
    for seq in features_list:
        seq_len = seq.shape[0]
        if seq_len < max_seq_len:
            # Pad with zeros
            pad_size = (max_seq_len - seq_len, n_features, frames_per_slice)
            padding = torch.zeros(pad_size)
            seq = torch.cat([seq, padding], dim=0)
            mask = torch.tensor([1]*seq_len + [0]*(max_seq_len - seq_len), dtype=torch.bool)
        else:
            mask = torch.ones(seq_len, dtype=torch.bool)
        padded_features.append(seq)
        feature_masks.append(mask)
    batch_features = torch.stack(padded_features)    # Shape: [batch_size, max_seq_len, n_features, frames_per_slice]
    batch_feature_masks = torch.stack(feature_masks)  # Shape: [batch_size, max_seq_len]

    # Pad labels
    max_num_labels = 13
    padded_labels = []
    label_masks = []
    for labels in labels_list:
        num_labels = len(labels)
        if num_labels < max_num_labels:
            # Pad with "no note" labels
            padding = [torch.tensor([-1.0]*5, dtype=torch.float32) for _ in range(max_num_labels - num_labels)]
            labels.extend(padding)
            mask = torch.tensor([1]*num_labels + [0]*(max_num_labels - num_labels), dtype=torch.bool)
        else:
            mask = torch.ones(num_labels, dtype=torch.bool)
        padded_labels.append(labels)
        label_masks.append(mask)
    batch_labels = torch.stack(padded_labels)     # Shape: [batch_size, max_num_labels, 5]
    batch_label_masks = torch.stack(label_masks)  # Shape: [batch_size, max_num_labels]

    return {
        'features': batch_features,
        'feature_masks': batch_feature_masks,
        'labels': batch_labels,
        'label_masks': batch_label_masks
    }

import torch
from collections import Counter

def analyze_label_distribution(labels):
    """
    Analyzes and prints the distribution of each label in the dataset.
    
    Args:
        labels (torch.Tensor): Tensor of shape [num_samples, max_notes, num_attributes]
    """
    line_index_labels = [label[1].item() for label in labels]
    line_layer_labels = [label[2].item() for label in labels]
    note_type_labels = [label[3].item() for label in labels]
    cut_direction_labels = [label[4].item() for label in labels]

    
    print("Label Distribution:")
    print("LineIndex:", Counter(line_index_labels))
    print("LineLayer:", Counter(line_layer_labels))
    print("NoteType:", Counter(note_type_labels))
    print("CutDirection:", Counter(cut_direction_labels))
