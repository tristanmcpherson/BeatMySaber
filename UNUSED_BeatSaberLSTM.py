import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BeatSaberLSTMModel(pl.LightningModule):
    def __init__(self, n_features, time_steps, hidden_size=64, num_layers=1, learning_rate=1e-3):
        super(BeatSaberLSTMModel, self).__init__()
        self.learning_rate = learning_rate

        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate the output size after CNN layers
        cnn_output_size = 128 * time_steps  # Since Conv1d preserves the time_steps dimension

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output Layers
        self.fc_beat_time_offset = nn.Linear(hidden_size, 1)        # Regression output
        self.fc_lineIndex = nn.Linear(hidden_size, 4)               # Categorical (4 classes for lineIndex)
        self.fc_lineLayer = nn.Linear(hidden_size, 3)               # Categorical (3 classes for lineLayer)
        self.fc_note_type = nn.Linear(hidden_size, 4)               # Categorical (4 classes for note types) [0, 1, 3] and [2] for "no note"
        self.fc_cut_direction = nn.Linear(hidden_size, 9)           # Categorical (9 classes for cut directions)

    def forward(self, x):
        batch_size, seq_len, n_features, time_steps = x.size()
        # Reshape for CNN
        x = x.view(batch_size * seq_len, n_features, time_steps)  # Merge batch and seq_len
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back: [batch_size, seq_len, cnn_output_size]

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Get the output from the last time step
        lstm_out_last = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_size]

        # Output layers
        beat_time_offset = self.fc_beat_time_offset(lstm_out_last).squeeze(-1)  # [batch_size]
        lineIndex_logits = self.fc_lineIndex(lstm_out_last)                       # Logits for lineIndex
        lineLayer_logits = self.fc_lineLayer(lstm_out_last)                       # Logits for lineLayer
        note_type_logits = self.fc_note_type(lstm_out_last)
        cut_direction_logits = self.fc_cut_direction(lstm_out_last)

        return beat_time_offset, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits

    def training_step(self, batch, batch_idx):
        sequences, labels = batch  # sequences: [batch_size, seq_len, n_features, time_steps]
        sequences = sequences.float()
        labels = labels.float()

        # Forward pass
        outputs = self(sequences)
        beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits = outputs

        # Extract labels from the last time step
        beat_time_offset_true = labels[:, -1, 0]
        lineIndex_true = labels[:, -1, 1].long()
        lineLayer_true = labels[:, -1, 2].long()
        note_type_true = labels[:, -1, 3].long()
        cut_direction_true = labels[:, -1, 4].long()

        # Create a mask for valid notes (i.e., ignore "no note" cases)
        valid_mask = (note_type_true != 2)  # Shape: [batch_size]

        # Ensure the mask matches the batch size
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # Apply the mask to filter out invalid samples
        beat_time_offset_pred = beat_time_offset_pred[valid_mask]       # [num_valid]
        beat_time_offset_true = beat_time_offset_true[valid_mask]       # [num_valid]

        lineIndex_logits = lineIndex_logits[valid_mask]                 # [num_valid, 4]
        lineIndex_true = lineIndex_true[valid_mask]                     # [num_valid]

        lineLayer_logits = lineLayer_logits[valid_mask]                 # [num_valid, 3]
        lineLayer_true = lineLayer_true[valid_mask]                     # [num_valid]

        note_type_logits = note_type_logits[valid_mask]                 # [num_valid, 3]
        note_type_true = note_type_true[valid_mask]                     # [num_valid]

        cut_direction_logits = cut_direction_logits[valid_mask]         # [num_valid, 9]
        cut_direction_true = cut_direction_true[valid_mask]             # [num_valid]

        # Compute individual losses
        loss_time_offset = F.mse_loss(beat_time_offset_pred, beat_time_offset_true)
        loss_lineIndex = F.cross_entropy(lineIndex_logits, lineIndex_true)
        loss_lineLayer = F.cross_entropy(lineLayer_logits, lineLayer_true)
        loss_note_type = F.cross_entropy(note_type_logits, note_type_true)
        loss_cut_direction = F.cross_entropy(cut_direction_logits, cut_direction_true)

        # Calculate total loss
        total_loss = (
            loss_time_offset +
            loss_lineIndex +
            loss_lineLayer +
            loss_note_type +
            loss_cut_direction
        )

        # Log individual losses
        self.log('loss_time_offset', loss_time_offset, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_lineIndex', loss_lineIndex, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_lineLayer', loss_lineLayer, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_note_type', loss_note_type, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_cut_direction', loss_cut_direction, on_step=True, on_epoch=True, prog_bar=True)

        # Log total loss (only once)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
