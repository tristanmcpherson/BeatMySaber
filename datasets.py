import os
import json
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from AudioFeatureExtraction import AudioFeatureExtractor
from config import Config

class BeatSaberDataset(torch.utils.data.Dataset):
    def __init__(self, config: Config):
        self._load_config(config)
        self.song_files = self._gather_files()

        self.n_features = self.audio_feature_extractor.get_num_features()
        self.window_size = config.overlap_ms // 25 # Number of slices to look back and forward

        self.song_data = self._prepare_song_data()

        dataset_metadata = {
            'song_data': self.song_data,
            'n_features': self.n_features,
            'time_steps': self.time_steps
        }
        torch.save(dataset_metadata, os.path.join(self.data_dir, 'model.pth'))
    
    def _load_config(self, config: Config): 
        self.slice_duration = config.slice_duration_ms  # Duration in ms
        self.sample_rate = config.sample_rate
        self.data_dir = config.data_dir
        self.difficulty = config.difficulty
        self.seq_len = config.seq_len  # Sequence length for ???
        self.overlap_ms = config.overlap_ms  # Overlap duration in ms
        self.max_notes = config.max_notes
        self.chunk_duration_sec = config.chunk_duration_sec  # 30-second chunks
        self.time_steps = config.time_steps

        self.audio_feature_extractor = AudioFeatureExtractor(
            sample_rate=self.sample_rate,
            hop_length=config.hop_length
        )

    def load_metadata(self, config: Config):
        """Load pre-saved metadata from a .pth file."""
        self._load_config(config)
        metadata = torch.load(os.path.join(self.data_dir, 'model.pth'))
        self.song_data = metadata['song_data']
        self.n_features = metadata['n_features']
        self.time_steps = metadata['time_steps']



    def _gather_files(self):
        song_files = []
        for root, dirs, files in os.walk(self.data_dir):
            if 'song.ogg' in files and 'Info.dat' in files:
                info_dat_path = os.path.join(root, 'Info.dat')
                with open(info_dat_path, 'r') as f:
                    info_dat_content = json.load(f)
                bpm = info_dat_content['_beatsPerMinute']

                # Find the beatmap file for the selected difficulty
                beatmap_filename = None
                for beatmap_set in info_dat_content['_difficultyBeatmapSets']:
                    if beatmap_set['_beatmapCharacteristicName'] == 'Standard':
                        for beatmap in beatmap_set['_difficultyBeatmaps']:
                            if beatmap['_difficulty'] == self.difficulty:
                                beatmap_filename = beatmap['_beatmapFilename']
                                break
                if beatmap_filename:
                    song_files.append((
                        os.path.join(root, 'song.ogg'),
                        os.path.join(root, beatmap_filename),
                        bpm
                    ))
        return song_files
    
    def load_and_preprocess_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if sr != self.audio_feature_extractor.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.audio_feature_extractor.sample_rate)
            audio = resampler(audio)

        return audio.float()

    def create_sequences_with_context(self):
        """
        Creates sequences with context by padding the beginning and end of song slices and
        labels to include overlapping context from neighboring slices.

        Returns:
            A list of tuples containing (sequence_features, sequence_labels) with context.
                - sequence_features: Tensor of shape [2 * window_size + 1, n_features, time_steps]
                - sequence_labels: Tensor of shape [(2 * window_size + 1) * max_notes, 5]
        """
        processed = []

        # Iterate over each song in self.song_data (prepared in _prepare_song_data)
        for song_slices, song_labels in self.songs_data:
            num_slices = len(song_slices)
            pad_slice = torch.zeros_like(song_slices[0])  # [n_features, time_steps]

            # Create pad_label as [max_notes, 5]
            pad_label = torch.stack([self.get_no_note_label()] * self.max_notes)  # [max_notes, 5]

            # Pad the beginning and end of the song slices
            # Create padding tensors
            slice_padding_before = pad_slice.unsqueeze(0).repeat(self.window_size,
                                                                 1)  # [window_size, n_features, time_steps]
            slice_padding_after = pad_slice.unsqueeze(0).repeat(self.window_size, 1,
                                                                1)  # [window_size, n_features, time_steps]

            # Concatenate padding with song_slices
            # Convert song_slices from list to tensor: [num_slices, n_features, time_steps]
            song_slices_tensor = torch.stack(song_slices)  # [num_slices, n_features, time_steps]
            padded_slices = torch.cat([slice_padding_before, song_slices_tensor, slice_padding_after],
                                      dim=0)  # [num_slices + 2 * window_size, n_features, time_steps]

            # Pad the labels
            # Convert song_labels from list to tensor: [num_slices, max_notes, 5]
            song_labels_tensor = torch.stack(song_labels)  # [num_slices, max_notes, 5]
            label_padding_before = pad_label.unsqueeze(0).repeat(self.window_size, 1, 1)  # [window_size, max_notes, 5]
            label_padding_after = pad_label.unsqueeze(0).repeat(self.window_size, 1, 1)  # [window_size, max_notes, 5]
            padded_labels = torch.cat([label_padding_before, song_labels_tensor, label_padding_after],
                                      dim=0)  # [num_slices + 2 * window_size, max_notes, 5]

            # Iterate through each slice and create a window of context slices around it.
            for i in range(self.window_size, self.window_size + num_slices):
                # Extract a window of slices (context) around the current slice.
                window_slices = padded_slices[
                                i - self.window_size: i + self.window_size + 1]  # [2 * window_size + 1, n_features, time_steps]
                window_labels = padded_labels[
                                i - self.window_size: i + self.window_size + 1]  # [2 * window_size + 1, max_notes, 5]

                # Sequence features
                sequence_features = window_slices  # [2 * window_size + 1, n_features, time_steps]

                # Sequence labels: flatten across window and notes
                sequence_labels = window_labels.view(-1, 5)  # [(2 * window_size +1) * max_notes, 5]

                # Append to processed
                processed.append((sequence_features, sequence_labels))

        return processed

    def beat_to_time(self, beat, bpm):
        """Convert beats to seconds based on BPM."""
        return beat * (60.0 / bpm)

    def encode_note_properties(self, note, bpm):
        """Encode the properties of a note into a tensor."""
        slice_duration_sec = self.slice_duration / 1000
        note_time_sec = self.beat_to_time(note['_time'], bpm)
        beat_time_offset = (note_time_sec % slice_duration_sec) / slice_duration_sec  # Normalize to [0, 1]

        return torch.tensor([
            beat_time_offset,
            note['_lineIndex'],    # lineIndex is the column, [-1](0, 3)
            note['_lineLayer'],    # lineLayer is the row, [-1](0, 2)
            note['_type'],         # Note type [-1](0, 1, 3)
            note['_cutDirection']      # Cut direction [-1](0 to 8)
        ], dtype=torch.float32)

    def _prepare_song_data(self):
        """
        Prepares the song data by splitting it into 30-second chunks, loading audio, extracting features,
        and aligning Beat Saber notes with these chunks.

        Returns:
            List of tuples: Each tuple contains (chunk_features, chunk_labels)
                - chunk_features: Tensor of shape [n_features, time_steps]
                - chunk_labels: List of encoded labels, length == max_notes
        """
        song_data = []
        chunk_samples = int(self.chunk_duration_sec * self.sample_rate / self.audio_feature_extractor.hop_length)
        max_notes = self.max_notes  # e.g., 13

        for audio_path, dat_path, bpm in self.song_files:
            # Load and preprocess audio (entire song)
            audio = self.load_and_preprocess_audio(audio_path)

            # Use the AudioFeatureExtractor to get features for the whole song
            song_features = self.audio_feature_extractor(audio)  # Expected shape: [n_features, total_frames]

            # Calculate the total number of chunks
            total_frames = song_features.shape[-1]
            num_chunks = (total_frames + chunk_samples - 1) // chunk_samples  # Ceiling division

            # Load the Beat Saber map (notes)
            with open(dat_path, 'r') as f:
                dat_content = json.load(f)
            notes = dat_content['_notes']

            # Sort notes by time to ensure correct alignment
            notes_sorted = sorted(notes, key=lambda x: x['_time'])

            # Iterate over chunks
            for i in range(num_chunks):
                start_frame = i * chunk_samples
                end_frame = start_frame + chunk_samples

                # Extract chunk features
                chunk_features = song_features[:, start_frame:end_frame]  # [n_features, chunk_samples]

                # Ensure consistent chunk length by padding the last chunk if necessary
                if chunk_features.shape[1] < chunk_samples:
                    padding = chunk_samples - chunk_features.shape[1]
                    chunk_features = F.pad(chunk_features, (0, padding), "constant", 0)  # [n_features, chunk_samples]

                # Convert frame indices to time in seconds
                chunk_start_time = librosa.frames_to_time(
                    start_frame, sr=self.sample_rate, hop_length=self.audio_feature_extractor.hop_length
                )
                chunk_end_time = librosa.frames_to_time(
                    end_frame, sr=self.sample_rate, hop_length=self.audio_feature_extractor.hop_length
                )

                # Collect all notes in the current chunk
                chunk_notes = [
                    note for note in notes_sorted
                    if chunk_start_time <= self.beat_to_time(note['_time'], bpm) < chunk_end_time
                ]

                # Encode labels
                chunk_labels = [self.encode_note_properties(note, bpm) for note in chunk_notes]

                # Handle padding if number of labels is less than max_notes
                num_labels = len(chunk_labels)
                if num_labels < max_notes:
                    num_pads = max_notes - num_labels
                    pad_label = self.get_no_note_label()  # [5]
                    chunk_labels.extend([pad_label] * num_pads)
                elif num_labels > max_notes:
                    # Truncate excess labels
                    chunk_labels = chunk_labels[:max_notes]

                # If there are no labels, pad with "no note" labels
                if not chunk_labels:
                    chunk_labels = [self.get_no_note_label() for _ in range(max_notes)]

                # Ensure chunk_features has shape [n_features, time_steps]
                assert chunk_features.shape == (self.n_features, chunk_samples), (
                    f"Expected chunk_features shape {(self.n_features, chunk_samples)}, got {chunk_features.shape}"
                )

                # Append to song_data
                song_data.append((chunk_features, chunk_labels))

        return song_data

    def get_no_note_label(self):
        """Returns the default 'no note' label with -1.0 for all fields."""
        return torch.tensor([
            -1.0,  # beat_time_offset
            -1.0,  # lineIndex
            -1.0,  # lineLayer
            -1.0,  # note_type
            -1.0  # cut_direction
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.song_data)

    # In your Dataset class
    def __getitem__(self, idx):
        chunk_features, chunk_labels = self.song_data[idx]
        return {
            'features': chunk_features,  # [n_features, time_steps]
            'labels': torch.stack(chunk_labels)  # [max_notes, 5]
        }
