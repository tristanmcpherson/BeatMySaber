import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram
import librosa
import numpy as np
import os
import re
import json
from torch.utils.data import DataLoader

# AudioFeatureExtraction.py

import torch
import torchaudio
import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100, hop_length=512, n_mfcc=13, n_chroma=12):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        
        # Define MFCC transform using torchaudio
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': 2048,
                'hop_length': self.hop_length,
                'n_mels': 128,
                'center': True,
                'pad_mode': 'reflect',
                'power': 2.0
            }
        )
        
    def __call__(self, audio):
        """
        Extracts MFCC and Chroma features from audio.
        
        Args:
            audio (Tensor): Tensor of shape [channels, samples]
        
        Returns:
            Tensor: Concatenated MFCC and Chroma features of shape [n_features, time_steps]
        """
        # Ensure audio is mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Extract MFCCs using torchaudio
        mfcc = self.mfcc_transform(audio)  # Shape: [n_mfcc, time_steps]
        mfcc = mfcc.squeeze(0).numpy()     # Convert to NumPy array [n_mfcc, time_steps]
        
        # Extract Chroma Features using librosa
        audio_np = audio.squeeze(0).numpy()  # [samples]
        chroma = librosa.feature.chroma_stft(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )  # Shape: [n_chroma, time_steps]
        
        # Ensure chroma and mfcc have the same number of time steps
        min_time_steps = min(mfcc.shape[1], chroma.shape[1])
        mfcc = mfcc[:, :min_time_steps]
        chroma = chroma[:, :min_time_steps]
        
        # Concatenate MFCC and Chroma features
        features = np.concatenate((mfcc, chroma), axis=0)  # Shape: [n_mfcc + n_chroma, time_steps]
        
        # Convert to PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        return features_tensor  # [n_features, time_steps]
    
    def get_num_features(self):
        """Returns the total number of features."""
        return self.n_mfcc + self.n_chroma
