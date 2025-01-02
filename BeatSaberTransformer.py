# BeatSaberTransformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class BeatSaberTransformerModel(pl.LightningModule):
    def __init__(self,
                 n_features,
                 hidden_size=128,
                 num_layers=4,
                 nhead=8,
                 dropout=0.1,
                 learning_rate=1e-4,
                 max_notes=13,
                 num_classes=None,
                 **kwargs  # Add **kwargs to capture any additional arguments
                 ):
        super(BeatSaberTransformerModel, self).__init__()

        # Avoid mutable default arguments by setting num_classes inside the method
        if num_classes is None:
            num_classes = {
                'lineIndex': 4,
                'lineLayer': 3,
                'note_type': 2,
                'cut_direction': 9
            }
        self.num_classes = num_classes  # Assign num_classes to an instance variable

        # Save hyperparameters explicitly to avoid issues with **kwargs
        self.save_hyperparameters('n_features', 'hidden_size', 'num_layers', 'nhead', 'dropout', 'learning_rate',
                                  'max_notes', 'num_classes')
        self.learning_rate = learning_rate
        self.max_notes = max_notes

        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
            # You may add pooling layers if necessary
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=hidden_size)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output Layers
        self.fc_beat_time_offset = nn.Linear(hidden_size, max_notes)  # Regression output per note

        self.fc_lineIndex = nn.Linear(hidden_size, max_notes * num_classes['lineIndex'])
        self.fc_lineLayer = nn.Linear(hidden_size, max_notes * num_classes['lineLayer'])
        self.fc_note_type = nn.Linear(hidden_size, max_notes * num_classes['note_type'])
        self.fc_cut_direction = nn.Linear(hidden_size, max_notes * num_classes['cut_direction'])

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, max_seq_len, n_features, frames_per_slice]
            src_key_padding_mask: Tensor of shape [batch_size, max_seq_len]
        """
        batch_size, n_features, frames_per_slice = x.size()
        # Current shape: [batch_size, n_features, seq_len] = [16, 25, 2583]
        #x = x.permute(0, 2, 1)  # New shape: [batch_size, seq_len, n_features] = [16, 2583, 25]
        # Permute to [batch_size, frames_per_slice, n_features]
        x = x.permute(0, 2, 1)  # [16, 2583, 25]

        # Reshape to combine batch_size and seq_len for CNN processing
        x = x.contiguous().view(batch_size * frames_per_slice, n_features, 1)  # Example: [16*2583, 25, 1]

        x = self.cnn(x)  # [batch_size * seq_len, hidden_size, frames_per_slice]

        # Global average pooling over frames_per_slice dimension
        x = torch.mean(x, dim=2)  # [batch_size * seq_len, hidden_size]
        #x = x.permute(1, 0)
        #
        # # Reshape back to [batch_size, seq_len, hidden_size]
        # x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]

        # Reshape back to [batch_size, frames_per_slice, hidden_size]
        x = x.view(batch_size, frames_per_slice, -1)  # [16, 2583, hidden_size]

        # Permute for Transformer [seq_len, batch_size, hidden_size]
        x = x.permute(1, 0, 2)  # [2583, 16, hidden_size]
        # Apply positional encoding
        x = self.pos_encoder(x)  # [batch_size, seq_len, hidden_size]

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=None)  # [batch_size, seq_len, hidden_size]

        # Aggregate sequence information (e.g., by averaging over the sequence length)
        x = torch.mean(x, dim=1)  # [batch_size, hidden_size]

        # Generate outputs
        beat_time_offset_pred = self.fc_beat_time_offset(x)  # [batch_size, max_notes]

        # Generate classification logits and reshape to [batch_size, max_notes, num_classes]
        lineIndex_logits = self.fc_lineIndex(x).view(batch_size, self.max_notes, -1)  # [batch_size, max_notes, 4]
        lineLayer_logits = self.fc_lineLayer(x).view(batch_size, self.max_notes, -1)  # [batch_size, max_notes, 3]
        note_type_logits = self.fc_note_type(x).view(batch_size, self.max_notes, -1)  # [batch_size, max_notes, 2]
        cut_direction_logits = self.fc_cut_direction(x).view(batch_size, self.max_notes, -1)  # [batch_size, max_notes, 9]

        return beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits

    def training_step(self, batch, batch_idx):
        features = batch['features']  # [batch_size, seq_len, n_features, frames_per_slice]
        feature_masks = batch['feature_masks']  # [batch_size, seq_len]
        labels = batch['labels']  # [batch_size, max_num_labels, 5]
        label_masks = batch['label_masks']  # [batch_size, max_num_labels]

        # Move tensors to the appropriate device
        features = features.to(self.device)
        feature_masks = feature_masks.to(self.device)
        labels = labels.to(self.device)
        label_masks = label_masks.to(self.device)

        # Invert feature_masks to create src_key_padding_mask (True for padded positions)
        src_key_padding_mask = ~feature_masks  # [batch_size, seq_len]

        # Forward pass through the model
        beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits = self(
            features,
            src_key_padding_mask=src_key_padding_mask
        )
        # Outputs:
        # beat_time_offset_pred: [batch_size, max_num_labels]
        # lineIndex_logits: [batch_size, max_num_labels, num_classes_lineIndex]
        # lineLayer_logits: [batch_size, max_num_labels, num_classes_lineLayer]
        # note_type_logits: [batch_size, max_num_labels, num_classes_note_type]
        # cut_direction_logits: [batch_size, max_num_labels, num_classes_cut_direction]

        # Flatten labels and masks
        batch_size, max_num_labels, _ = labels.size()
        valid_mask = label_masks.view(batch_size * max_num_labels)  # [batch_size * max_num_labels]
        labels_flat = labels.view(batch_size * max_num_labels, 5)  # [batch_size * max_num_labels, 5]

        # Extract true labels
        beat_time_offset_true = labels_flat[:, 0]  # [batch_size * max_num_labels]
        lineIndex_true = labels_flat[:, 1].long()  # [batch_size * max_num_labels]
        lineLayer_true = labels_flat[:, 2].long()
        note_type_true = labels_flat[:, 3].long()
        cut_direction_true = labels_flat[:, 4].long()

        # Flatten predictions
        beat_time_offset_pred = beat_time_offset_pred.view(-1)  # [batch_size * max_num_labels]
        lineIndex_logits = lineIndex_logits.view(-1, self.hparams.num_classes['lineIndex'])
        lineLayer_logits = lineLayer_logits.view(-1, self.hparams.num_classes['lineLayer'])
        note_type_logits = note_type_logits.view(-1, self.hparams.num_classes['note_type'])
        cut_direction_logits = cut_direction_logits.view(-1, self.hparams.num_classes['cut_direction'])

        # Apply valid_mask to predictions and labels
        beat_time_offset_pred = beat_time_offset_pred[valid_mask]
        beat_time_offset_true = beat_time_offset_true[valid_mask]

        lineIndex_logits = lineIndex_logits[valid_mask]
        lineIndex_true = lineIndex_true[valid_mask]

        lineLayer_logits = lineLayer_logits[valid_mask]
        lineLayer_true = lineLayer_true[valid_mask]

        note_type_logits = note_type_logits[valid_mask]
        note_type_true = note_type_true[valid_mask]

        cut_direction_logits = cut_direction_logits[valid_mask]
        cut_direction_true = cut_direction_true[valid_mask]

        # Compute losses
        loss_time_offset = F.mse_loss(beat_time_offset_pred, beat_time_offset_true)

        loss_lineIndex = F.cross_entropy(lineIndex_logits, lineIndex_true)
        loss_lineLayer = F.cross_entropy(lineLayer_logits, lineLayer_true)
        loss_note_type = F.cross_entropy(note_type_logits, note_type_true)
        loss_cut_direction = F.cross_entropy(cut_direction_logits, cut_direction_true)

        # Total loss
        total_loss = loss_time_offset + loss_lineIndex + loss_lineLayer + loss_note_type + loss_cut_direction

        # Logging losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer