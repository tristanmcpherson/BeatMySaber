import torch
from torch.nn.utils.rnn import pad_sequence

from config import Config

config = Config()

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and labels.
    Args:
        batch: List of dictionaries containing 'features' and 'labels'
    """
    # Extract features and labels
    features = [item['features'] for item in batch]  # List of [n_features, time_steps]
    labels = [item['labels'] for item in batch]      # List of [max_notes, 5]
    
    # Get max time steps in batch
    max_time_steps = max(feat.size(1) for feat in features)
    
    # Pad features and create masks
    padded_features = []
    feature_masks = []
    
    for feat in features:
        n_features, time_steps = feat.size()
        padding_size = max_time_steps - time_steps
        
        if padding_size > 0:
            # Pad features
            padded_feat = F.pad(feat, (0, padding_size), "constant", 0)
            # Create mask (1 for real, 0 for padding)
            mask = torch.ones(max_time_steps, dtype=torch.bool)
            mask[time_steps:] = False
        else:
            padded_feat = feat
            mask = torch.ones(max_time_steps, dtype=torch.bool)
            
        padded_features.append(padded_feat)
        feature_masks.append(mask)
    
    # Stack everything
    features_tensor = torch.stack(padded_features)  # [batch_size, n_features, max_time_steps]
    feature_masks = torch.stack(feature_masks)      # [batch_size, max_time_steps]
    labels_tensor = torch.stack(labels)             # [batch_size, max_notes, 5]
    
    return {
        'features': features_tensor,
        'feature_masks': feature_masks,
        'labels': labels_tensor,
        'label_masks': (labels_tensor[:, :, 0] != -1)  # Create mask based on beat_time_offset
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
