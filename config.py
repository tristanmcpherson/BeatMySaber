# config.py

from dataclasses import dataclass

@dataclass
class Config:
    # Dataset Parameters
    data_dir: str = './beatmaps/'
    output_path: str = './output/dataset.pth'
    difficulty: str = 'ExpertPlus'
    slice_duration_ms: int = 250  # Duration in milliseconds
    sample_rate: int = 44100
    seq_len: int = 16  # Sequence length for LSTM
    max_notes: int = 13
    chunk_duration_sec: int = 30
    
    # Audio Feature Extraction
    hop_length: int = 512  # Must match AudioFeatureExtractor's hop_length
    
    # Model Hyperparameters
    n_features: int = 25  # To be dynamically determined if possible
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-4
    time_steps: int = 0  # To be set based on dataset
    
    # Training Parameters
    batch_size: int = 16
    num_workers: int = 1
    max_epochs: int = 500

    overlap_ms: int = 1000
    
    # Logging
    log_dir: str = 'lightning_logs'
    model_checkpoint_dir: str = 'checkpoints/'

    def __init__(self):
        self.time_steps = self._calculate_time_steps()


    def _calculate_time_steps(self):
        """Calculate time_steps based on slice_duration and AudioFeatureExtractor's hop_length."""
        time_steps = int(
            self.slice_duration_ms * self.sample_rate / self.hop_length
        )
        return time_steps