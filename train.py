from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import BeatSaberDataset
from models import BeatSaberCNN
from utils import collate_fn, create_attention_mask

def train():
    # Directory containing song.ogg and corresponding beatmap.dat files
    data_dir = './beatmaps/'
    
    # Initialize the dataset
    dataset = BeatSaberDataset(data_dir)

    # Assuming dataset returns (src, tgt, mask)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = BeatSaberCNN()

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    train()
