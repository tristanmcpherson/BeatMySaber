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

class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=44100, n_mfcc=13, n_chroma=12, n_fft=2048, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.mfcc_transform = MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': 40}
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    def forward(self, audio):
        # Ensure audio is float32 and mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        mfcc = self.mfcc_transform(audio)
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()  # Normalize MFCC

        mel_specgram = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_chroma
        )(audio)
        chroma = librosa.feature.chroma_cqt(C=mel_specgram.numpy(), sr=self.sample_rate)
        chroma = torch.from_numpy(chroma).float()
        chroma = (chroma - chroma.mean()) / chroma.std()  # Normalize Chroma

        return torch.cat([mfcc, chroma], dim=1)
