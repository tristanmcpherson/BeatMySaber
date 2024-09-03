import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import librosa
import json
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset
import librosa
import json
import os

torch.set_float32_matmul_precision('high')  # or 'medium' depending on your preference


class BeatSaberDataset(Dataset):
    def __init__(self, data_dir, difficulty='Expert', sample_rate=22050):
        """
        Args:
            data_dir (str): Directory containing subdirectories with song.ogg and difficulty .dat pairs.
            difficulty (str): Difficulty level to load (e.g., 'Expert', 'Hard', 'Normal').
            sample_rate (int): The sample rate for loading audio.
        """
        self.data_dir = data_dir
        self.difficulty = difficulty
        self.sample_rate = sample_rate
        self.song_files = self._gather_files()
        
        # Debugging: Print the number of song files found
        print(f"Found {len(self.song_files)} song files with difficulty '{difficulty}' in {data_dir}")
    
    def _gather_files(self):
        """Recursively find all song.ogg files and their corresponding difficulty .dat files."""
        song_files = []
        for root, dirs, files in os.walk(self.data_dir):
            song_file = 'song.ogg'
            beatmap_file = f'{self.difficulty}.dat'
            if song_file in files and beatmap_file in files:
                song_path = os.path.join(root, song_file)
                beatmap_path = os.path.join(root, beatmap_file)
                song_files.append((song_path, beatmap_path))
        return song_files
    
    def __len__(self):
        return len(self.song_files)


    def __getitem__(self, idx):
        song_path, beatmap_path = self.song_files[idx]
        
        # Load and process audio
        features, _ = self._process_audio(song_path)
        length = features.shape[0]  # Assuming you are returning the sequence length

        # Load beatmap
        labels = self._load_beatmap(beatmap_path)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), length


    def _process_audio(self, file_path):
        """Load audio file and extract features."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        features = np.concatenate((mfcc, chroma, spectral_contrast), axis=0)
        return features.T, sr  # Transpose for correct shape

    def _load_beatmap(self, file_path):
        """Load the beatmap file and parse notes."""
        with open(file_path, 'r') as f:
            beatmap = json.load(f)
        
        notes = beatmap.get('_notes', [])
        labels = self._parse_notes(notes)
        return labels

    def _parse_notes(self, notes):
        """Convert note data to labels."""
        labels = []
        for note in notes:
            label = self._map_note_to_label(note)
            labels.append(label)
        return labels
    
    def _map_note_to_label(self, note):
        """Map note fields to a label."""
        label = (note['_cutDirection'] * 4) + note['_lineIndex']  # Example mapping
        return label
    
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence

class MultiTransformerModel(pl.LightningModule):
    def __init__(self, mfcc_dim, chroma_dim, contrast_dim, output_dim, nhead, nhid, nlayers):
        super(MultiTransformerModel, self).__init__()
        
        # Define transformer encoder layers with batch_first=True for ease of use
        encoder_layer_mfcc = nn.TransformerEncoderLayer(d_model=mfcc_dim, nhead=nhead, dim_feedforward=nhid, batch_first=True)
        self.mfcc_transformer = nn.TransformerEncoder(encoder_layer_mfcc, num_layers=nlayers)

        encoder_layer_chroma = nn.TransformerEncoderLayer(d_model=chroma_dim, nhead=nhead, dim_feedforward=nhid, batch_first=True)
        self.chroma_transformer = nn.TransformerEncoder(encoder_layer_chroma, num_layers=nlayers)

        encoder_layer_contrast = nn.TransformerEncoderLayer(d_model=contrast_dim, nhead=nhead, dim_feedforward=nhid, batch_first=True)
        self.contrast_transformer = nn.TransformerEncoder(encoder_layer_contrast, num_layers=nlayers)
        
        # Final fully connected layer
        combined_dim = mfcc_dim + chroma_dim + contrast_dim
        self.fc = nn.Linear(combined_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        features, labels, length = batch  # Unpack all three elements
        output = self(features)   # Pass lengths to the forward method if needed
        
        # Assuming labels are padded with -100 as ignored index
        loss = self.loss_fn(output.view(-1, output.shape[-1]), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def forward(self, src):
        # Split input features into their respective groups
        mfcc_features = src[:, :, :13]  # Assuming first 13 dims are MFCC
        chroma_features = src[:, :, 13:25]  # Next 12 dims are Chroma
        contrast_features = src[:, :, 25:]  # Remaining dims are Spectral Contrast

        # Forward pass through each transformer
        mfcc_output = self.mfcc_transformer(mfcc_features)
        chroma_output = self.chroma_transformer(chroma_features)
        contrast_output = self.contrast_transformer(contrast_features)

        # Concatenate the outputs along the feature dimension
        combined_output = torch.cat((mfcc_output, chroma_output, contrast_output), dim=2)

        # Pass the combined output through the final fully connected layer
        output = self.fc(combined_output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def collate_fn(batch):
    """Custom collate function to pad sequences."""
    features, labels, lengths = zip(*batch)  # Unpack three elements

    # Sort by sequence length (optional, but often useful for RNNs/Transformers)
    lengths, sorted_idx = torch.tensor(lengths).sort(0, descending=True)
    features = [features[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    # Pad features and labels to have the same length
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored labels

    return features_padded, labels_padded, lengths


# Example dimensions for input feature sets
mfcc_dim = 13
chroma_dim = 12
contrast_dim = 14
output_dim = 5  # Example output dimension

# Directory containing song.ogg and corresponding beatmap.dat files
data_dir = './beatmaps/'

# Initialize the dataset
dataset = BeatSaberDataset(data_dir)

print(dataset.song_files)

# Create a DataLoader for batching
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model = MultiTransformerModel(mfcc_dim, chroma_dim, contrast_dim, output_dim, nhead=1, nhid=512, nlayers=6)

# Train the model with PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=10)  # Set gpus=1 if using GPU
trainer.fit(model, train_dataloader)
