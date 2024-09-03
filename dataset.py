import torch
from torch.utils.data import Dataset
import librosa
import json
import os

class BeatSaberDataset(Dataset):
    def __init__(self, data_dir, sample_rate=22050):
        """
        Args:
            data_dir (str): Directory containing song and beatmap pairs.
            sample_rate (int): The sample rate for loading audio.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.song_files = [f for f in os.listdir(data_dir) if f.endswith('.ogg')]
    
    def __len__(self):
        return len(self.song_files)

    def __getitem__(self, idx):
        song_file = self.song_files[idx]
        beatmap_file = os.path.splitext(song_file)[0] + '.dat'

        # Load and process audio
        audio_path = os.path.join(self.data_dir, song_file)
        features, _ = self._process_audio(audio_path)

        # Load beatmap
        beatmap_path = os.path.join(self.data_dir, beatmap_file)
        labels = self._load_beatmap(beatmap_path)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

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
