import json
import math
import os
import re

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import torchaudio

from AudioFeatureExtraction import AudioFeatureExtractor
from utils import label_audio_frames_with_fractional_beats

class BeatSaberDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, difficulty='Expert', slice_duration=25, sample_rate=44100):
        self.slice_duration = slice_duration  # Duration of each slice in ms
        self.sample_rate = sample_rate
        self.data_dir = data_dir
        self.difficulty = difficulty
        self.song_files = self._gather_files()
        self.audio_feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

    def __len__(self):
        return len(self.song_files)

    def _gather_files(self):
        song_files = []
        for root, dirs, files in os.walk(self.data_dir):
            if 'song.ogg' in files and f'{self.difficulty}.dat' in files:
                bpm = self._extract_bpm_from_folder(root)  # Extract BPM from folder
                song_files.append((
                    os.path.join(root, 'song.ogg'),
                    os.path.join(root, f'{self.difficulty}.dat'),
                    bpm
                ))
        return song_files

    def __getitem__(self, idx):
        # Fetch the song data: audio file and map file
        audio_path, dat_path, bpm = self.song_files[idx]

        # Load and preprocess audio (entire song)
        audio, sr = torchaudio.load(audio_path)
        audio = audio.float()

        # Use the AudioFeatureExtractor to get features for the whole song
        song_features = self.audio_feature_extractor(audio)

        # Slice features into time slices
        slice_samples = int(self.slice_duration / 1000 * self.sample_rate)
        num_slices = song_features.shape[-1] // slice_samples

        features, labels = [], []

        # Load the Beat Saber map (notes)
        with open(dat_path, 'r') as f:
            dat_content = json.load(f)
        notes = dat_content['_notes']

        # Iterate over slices
        for i in range(num_slices):
            start_frame = i * slice_samples
            end_frame = min((i + 1) * slice_samples, song_features.shape[-1])

            slice_features = song_features[:, :, start_frame:end_frame]

            # Check if this slice contains any notes
            slice_start_time = librosa.frames_to_time(start_frame, sr=self.sample_rate)
            slice_end_time = librosa.frames_to_time(end_frame, sr=self.sample_rate)

            slice_notes = [note for note in notes if slice_start_time <= self.beat_to_time(note['_time'], bpm) < slice_end_time]

            if slice_notes:
                note = slice_notes[0]  # Handle the first note or apply merge logic
                note_label = self.encode_note_properties(note, bpm)
            else:
                note_label = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)  # No note

            features.append(slice_features)
            labels.append(note_label)

        return features, labels

    def beat_to_time(self, beat, bpm):
        """Convert beats to milliseconds based on BPM."""
        return beat / bpm * 60

    def encode_note_properties(self, note, bpm):
        """Encode the properties of a note into a tensor."""
        beat_time_offset = self.beat_to_time(note['_time'], bpm) % self.slice_duration

        return torch.tensor([
            beat_time_offset,
            note['_lineIndex'] / 3,  # Normalize index
            note['_lineLayer'] / 2,  # Normalize layer
            note['_type'],           # Note type (0 or 1)
            note['_cutDirection'] / 8  # Normalize cut direction
        ], dtype=torch.float32)

    def _extract_bpm_from_folder(self, folder_path):
        """Extract BPM from the folder name."""
        folder_name = os.path.basename(folder_path)
        match = re.search(r'\[(\d+)\]', folder_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("BPM not found in folder name")
 