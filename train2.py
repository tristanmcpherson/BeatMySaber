import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import os
import re
import json
import math
from torch.utils.data import DataLoader

class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=44100, n_mfcc=13, n_chroma=12, n_fft=2048, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, audio):
        # Ensure audio is float32
        audio = audio.float()

        # Ensure audio is mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_fft': self.n_fft, 'hop_length': self.hop_length}
        )(audio)
        
        audio_np = audio.numpy()[0]

        chroma = librosa.feature.chroma_stft(
            y=audio_np,
            sr=self.sample_rate,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        chroma = torch.from_numpy(chroma).float().unsqueeze(0)
        
        spec_cent = librosa.feature.spectral_centroid(
            y=audio_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spec_cent = torch.from_numpy(spec_cent).float().unsqueeze(0)
        
        # Concatenate features
        features = torch.cat([mfcc, chroma, spec_cent], dim=1)
        return features

class BeatSaberDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, difficulty='Expert', chunk_length=30):
        self.data_dir = data_dir
        self.difficulty = difficulty
        self.chunk_length = chunk_length  # in seconds
        self.song_files = self._gather_files()
        self.audio_feature_extractor = AudioFeatureExtractor()

    def __len__(self):
        return len(self.song_files)

    def __getitem__(self, idx):
        audio_path, dat_path, bpm = self.song_files[idx]
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.float()  # Ensure audio is float32
        
        # Load Beat Saber map
        with open(dat_path, 'r') as f:
            dat_content = json.load(f)
        
        notes = dat_content['_notes']
        notes = sorted(notes, key=lambda x: x['_time'])
        
        # Process audio and notes in chunks
        chunk_samples = self.chunk_length * sr
        num_chunks = math.ceil(audio.shape[1] / chunk_samples)
        
        chunks = []
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, audio.shape[1])
            
            # Extract audio features for this chunk
            chunk_audio = audio[:, start_sample:end_sample]
            chunk_features = self.audio_feature_extractor(chunk_audio)
            
            # Get notes for this chunk
            chunk_start_time = i * self.chunk_length
            chunk_end_time = (i + 1) * self.chunk_length
            chunk_notes = [note for note in notes if chunk_start_time <= note['_time'] < chunk_end_time]
            
            # Convert notes to tensor
            note_tensor = torch.zeros(len(chunk_notes), 5, dtype=torch.float32)
            for j, note in enumerate(chunk_notes):
                note_tensor[j] = torch.tensor([
                    (note['_time'] - chunk_start_time) / self.chunk_length,  # Normalize time within chunk
                    note['_lineIndex'] / 3,  # Normalize line index
                    note['_lineLayer'] / 2,  # Normalize line layer
                    note['_type'],
                    note['_cutDirection'] / 8  # Normalize cut direction
                ], dtype=torch.float32)
            
            chunks.append((chunk_features, note_tensor))
        
        return chunks

    def _gather_files(self):
        """Recursively find all song.ogg files and their corresponding difficulty .dat files."""
        song_files = []
        for root, dirs, files in os.walk(self.data_dir):
            if 'song.ogg' in files and f'{self.difficulty}.dat' in files:
                bpm = self._extract_bpm_from_folder(root)
                song_files.append((
                    os.path.join(root, 'song.ogg'),
                    os.path.join(root, f'{self.difficulty}.dat'),
                    bpm
                ))
        return song_files

    def _extract_bpm_from_folder(self, folder_path):
        """Extract BPM from the folder name using regex to find [BPM] pattern."""
        folder_name = os.path.basename(folder_path)
        match = re.search(r'\[(\d+)\]', folder_name)  # Look for [BPM] in the folder name
        if match:
            bpm = int(match.group(1))
            return bpm
        else:
            raise ValueError(f"BPM not found in folder name: {folder_name}")

class VariableLengthSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.feature_compression = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        print(f"Input src shape: {src.shape}")
        print(f"Input trg shape: {trg.shape}")
        
        if len(src.shape) != 3:
            raise ValueError(f"Expected src to have 3 dimensions, but got shape {src.shape}")
        
        batch_size, seq_len, feat_dim = src.shape
        
        # Reshape src to (batch_size * seq_len, feat_dim)
        src_reshaped = src.view(-1, feat_dim)
        
        # Compress the input features
        src_compressed = self.feature_compression(src_reshaped)
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        src_compressed = src_compressed.view(batch_size, seq_len, -1)

        outputs = torch.zeros(batch_size, trg.shape[1], self.fc_out.out_features, dtype=torch.float32, device=src.device)

        encoder_outputs, (hidden, cell) = self.encoder(src_compressed)

        input = trg[:, 0]

        for t in range(1, trg.shape[1]):
            output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
            output = self.fc_out(output.squeeze(1))
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs

def train_chunk(model, chunk_features, chunk_notes, optimizer, criterion, device):
    chunk_features, chunk_notes = chunk_features.to(device), chunk_notes.to(device)
    
    print(f"chunk_features shape before processing: {chunk_features.shape}")
    print(f"chunk_notes shape: {chunk_notes.shape}")
    
    # Ensure chunk_features is 3D: (batch_size, sequence_length, feature_dim)
    if chunk_features.dim() == 4:
        batch_size, channels, time_steps, features = chunk_features.shape
        chunk_features = chunk_features.permute(0, 2, 1, 3).reshape(batch_size, time_steps, channels * features)
    
    print(f"chunk_features shape after processing: {chunk_features.shape}")
    
    # Ensure data type is float32
    chunk_features = chunk_features.float()
    chunk_notes = chunk_notes.float()
    
    optimizer.zero_grad()
    output = model(chunk_features, chunk_notes)
    loss = criterion(output, chunk_notes)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        for chunk_features, chunk_notes in batch:
            loss = train_chunk(model, chunk_features.unsqueeze(0), chunk_notes.unsqueeze(0), optimizer, criterion, device)
            total_loss += loss
    return total_loss / len(train_loader)

def generate_map(model, audio_path, chunk_length=30, device='cpu'):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.float()  # Ensure audio is float32
    audio_feature_extractor = AudioFeatureExtractor()
    
    chunk_samples = chunk_length * sr
    num_chunks = math.ceil(audio.shape[1] / chunk_samples)
    
    generated_notes = []
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, audio.shape[1])
        
        chunk_audio = audio[:, start_sample:end_sample]
        chunk_features = audio_feature_extractor(chunk_audio).unsqueeze(0).to(device)
        
        # Reshape chunk_features to (batch_size, sequence_length, feature_dim)
        batch_size, channels, time_steps, features = chunk_features.shape
        chunk_features = chunk_features.permute(0, 2, 1, 3).reshape(batch_size, time_steps, channels * features)
        
        with torch.no_grad():
            output = model(chunk_features, torch.zeros(1, 1, 5, dtype=torch.float32, device=device))
        
        for note in output[0]:
            time = (note[0].item() + i) * chunk_length
            line_index = round(note[1].item() * 3)
            line_layer = round(note[2].item() * 2)
            note_type = round(note[3].item())
            cut_direction = round(note[4].item() * 8)
            
            generated_notes.append({
                "_time": time,
                "_lineIndex": line_index,
                "_lineLayer": line_layer,
                "_type": note_type,
                "_cutDirection": cut_direction
            })
    
    return generated_notes

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set hyperparameters
    data_dir = './beatmaps/'
    chunk_length = 30
    hidden_dim = 128
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 1  # We're using 1 because each song might have a different number of chunks

    # Create dataset and dataloader
    dataset = BeatSaberDataset(data_dir, chunk_length=chunk_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Calculate input_dim based on the actual size of the features
    sample_chunk = next(iter(train_loader))[0][0]  # Get the first chunk of the first batch
    input_dim = sample_chunk.shape[1] * sample_chunk.shape[2]  # channels * features

    print(f"Input dimension: {input_dim}")

    # Create model
    model = VariableLengthSeq2Seq(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=5, num_layers=num_layers).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure model parameters are float32
    for param in model.parameters():
        param.data = param.data.float()

    # Training loop
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'beat_saber_model.pth')

    # Generate a map for a new song
    new_song_path = "path/to/new_song.ogg"
    beat_saber_map = generate_map(model, new_song_path, device=device)

    # Save the generated map
    with open('generated_map.json', 'w') as f:
        json.dump(beat_saber_map, f, indent=2)

    print("Map generation complete. Check 'generated_map.json' for the result.")