import torch
from torch.nn.utils.rnn import pad_sequence


def create_attention_mask(labels, pad_token_id=-100):
    """
    Create an attention mask based on the labels.
    The mask is 1 where notes are present and 0 where no notes (or padding).
    """
    mask = (labels != pad_token_id).int()
    return mask

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    """
    features, labels = zip(*batch)


    flattened_features = [feat.view(feat.size(0) * feat.size(1), -1) for feat in features]

    # Pad features and labels to the same length within the batch
    # Using pad_sequence to handle sequences of different lengths
    features_padded = pad_sequence([torch.cat(f) for f in flattened_features], batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence([torch.cat(l) for l in labels], batch_first=True, padding_value=-100)

    return features_padded, labels_padded

def label_audio_frames_with_fractional_beats(features, beat_times, note_labels, bpm, feature_window_size, tolerance=0.01):
    """
    Label each audio feature frame with Beat Saber note information based on fractional beat timings.
    
    :param features: The audio features (MFCC, chroma, etc.).
    :param beat_times: List of Beat Saber beat times (_time values).
    :param note_labels: Corresponding note labels (e.g., _cutDirection, _lineIndex).
    :param bpm: Beats per minute of the song.
    :param feature_window_size: Time window size for each feature frame (e.g., 0.025 seconds).
    :param tolerance: Time tolerance (in beats) for matching notes to frames.
    :return: List of labels for each audio frame.
    """
    # Time per beat in seconds
    seconds_per_beat = 60 / bpm
    
    # Initialize frame labels with -1 (indicating no note)
    frame_labels = [-1] * len(features)
    
    # Loop through the beats and assign labels to the closest audio frames
    for beat_time, note_label in zip(beat_times, note_labels):
        # Convert beat time to seconds
        time_in_seconds = beat_time * seconds_per_beat
        
        # Convert time in seconds to frame index
        frame_idx = int(round(time_in_seconds / feature_window_size))
        
        # Ensure frame index is within bounds of the feature sequence
        if 0 <= frame_idx < len(features):
            frame_labels[frame_idx] = note_label
    
    return frame_labels

