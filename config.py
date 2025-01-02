# config.py

from dataclasses import dataclass

@dataclass
class Config:
    # Dataset Parameters
    data_dir: str = './beatmaps/'
    output_path: str = './output/dataset.pth'
    difficulty: str = 'ExpertPlus'
    slice_duration_ms: int = 500
    sample_rate: int = 44100
    seq_len: int = 1  # Changed to 1 since we're using transformer
    max_notes: int = 8
    chunk_duration_sec: int = 30
    
    # Audio Feature Extraction
    hop_length: int = 512
    
    # Model Hyperparameters
    n_features: int = 25
    hidden_size: int = 256
    num_layers: int = 3
    learning_rate: float = 1e-4
    
    # Training Parameters
    batch_size: int = 32
    num_workers: int = 6
    max_epochs: int = 500
    overlap_ms: int = 500
    
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