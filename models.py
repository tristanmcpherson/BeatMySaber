import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BeatSaberCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the architecture
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 50, 512)  # Assuming flattened size is 128*50, adjust based on your input size
        self.fc2 = nn.Linear(512, 6)  # Adjust based on your output requirements

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

