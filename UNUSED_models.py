import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from UNUSED_BeatSaberCNN import BeatSaberCNN

class BeatSaberLightningModel(pl.LightningModule):
    def __init__(self, n_features, time_steps):
        super().__init__()
        self.model = BeatSaberCNN(n_features, time_steps)
        self.learning_rate = 1e-3


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        features = features.float()
        labels = labels.float()

        outputs = self(features)
        
        # Split outputs
        beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits = outputs
        
        beat_time_offset_true = labels[:, 0]
        lineIndex_true = labels[:, 1].long()
        lineLayer_true = labels[:, 2].long()
        note_type_true = labels[:, 3].long()
        cut_direction_true = labels[:, 4].long()


        # Create masks for valid data (where targets are not the special "no note" value)
        valid_mask_regression = (beat_time_offset_true != -1.0)
        
        if valid_mask_regression.any():
            # Apply masks
            beat_time_offset_pred_valid = beat_time_offset_pred[valid_mask_regression]
            beat_time_offset_true_valid = beat_time_offset_true[valid_mask_regression]

            # Compute regression losses
            loss_time_offset = F.mse_loss(beat_time_offset_pred_valid, beat_time_offset_true_valid)
        else:
            # No valid regression data in this batch
            loss_time_offset = torch.tensor(0.0, requires_grad=True).to(self.device)

        # Losses
        loss_time_offset = F.mse_loss(beat_time_offset_pred, beat_time_offset_true)
        loss_lineIndex = F.cross_entropy(lineIndex_logits, lineIndex_true)
        loss_lineLayer = F.cross_entropy(lineLayer_logits, lineLayer_true)
        loss_note_type = F.cross_entropy(note_type_logits, note_type_true)
        loss_cut_direction = F.cross_entropy(cut_direction_logits, cut_direction_true)
        
        total_loss = loss_time_offset + loss_lineIndex + loss_lineLayer + loss_note_type + loss_cut_direction
        
        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer