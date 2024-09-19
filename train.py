# train.py

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from BeatSaberTransformer import BeatSaberTransformerModel
from datasets import BeatSaberDataset
from config import Config
from utils import collate_fn  # Ensure this is properly defined

torch.set_float32_matmul_precision("high")

def train():
    # Initialize Configuration
    config = Config()
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = BeatSaberDataset(config)
    
    # Update config with dynamic parameters
    config.n_features = dataset.n_features
    print(f'Dataset loaded with n_features={config.n_features}')
    
    # Create DataLoader
    print("Creating DataLoader...")
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # Ensure collate_fn correctly handles your data
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True
    )


    # Initialize model with dynamic parameters
    print("Initializing model...")
    model = BeatSaberTransformerModel(
        n_features=config.n_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        learning_rate=config.learning_rate
    )
    
    # Setup TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name='beat_saber_model'
    )
    
    # Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        save_top_k=1,
        mode='min',
        dirpath=config.model_checkpoint_dir,
        filename='beat_saber_model-{epoch:02d}-{train_loss:.4f}'
    )
    
    # Initialize PyTorch Lightning Trainer
    print("Training model...")
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='auto',  # Automatically use GPU if available
        devices=1,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    
    # Start training
    trainer.fit(model, train_dataloader)
    
    # Save the final model checkpoint
    trainer.save_checkpoint('beat_saber_model.ckpt')


if __name__ == '__main__':
    train()
