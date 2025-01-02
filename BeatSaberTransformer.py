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
                 max_notes=8,
                 num_classes=None,
                 **kwargs  # Add **kwargs to capture any additional arguments
                 ):
        super(BeatSaberTransformerModel, self).__init__()

        # Modify default num_classes to include bombs
        if num_classes is None:
            num_classes = {
                'lineIndex': 4,
                'lineLayer': 3,
                'note_type': 4,  # Changed from 2 to 4 to accommodate 0,1,3
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

        # Split the outputs into note presence and properties
        self.fc_note_presence = nn.Linear(hidden_size, max_notes)  # Binary classification for note presence
        self.fc_beat_time_offset = nn.Linear(hidden_size, max_notes)  # Only predict time for present notes
        self.fc_lineIndex = nn.Linear(hidden_size, max_notes * num_classes['lineIndex'])
        self.fc_lineLayer = nn.Linear(hidden_size, max_notes * num_classes['lineLayer'])
        self.fc_note_type = nn.Linear(hidden_size, max_notes * num_classes['note_type'])
        self.fc_cut_direction = nn.Linear(hidden_size, max_notes * num_classes['cut_direction'])

        # Add attention pooling layers
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        x: [batch_size, n_features, time_steps]
        """
        batch_size = x.size(0)
        
        # CNN processing
        x = self.cnn(x)  # [batch_size, hidden_size, time_steps]
        x = x.transpose(1, 2)  # [batch_size, time_steps, hidden_size]
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply attention pooling
        attention_weights = self.attention_pooling(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        x = torch.bmm(attention_weights.transpose(1, 2), x)  # [batch_size, 1, hidden_size]
        x = x.squeeze(1)  # [batch_size, hidden_size]

        # Generate outputs
        note_presence = torch.sigmoid(self.fc_note_presence(x))  # [batch_size, max_notes]
        beat_time_offset = self.fc_beat_time_offset(x)  # [batch_size, max_notes]
        
        # Reshape logits to [batch_size, max_notes, num_classes]
        lineIndex = self.fc_lineIndex(x).view(batch_size, self.max_notes, -1)
        lineLayer = self.fc_lineLayer(x).view(batch_size, self.max_notes, -1)
        note_type = self.fc_note_type(x).view(batch_size, self.max_notes, -1)
        cut_direction = self.fc_cut_direction(x).view(batch_size, self.max_notes, -1)

        return note_presence, beat_time_offset, lineIndex, lineLayer, note_type, cut_direction

    def training_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['labels']      # [32, 8, 5]
        label_masks = batch['label_masks']  # [32, 8]

        outputs = self(features)
        note_presence_pred, beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits = outputs

        # Flatten predictions and labels
        labels_flat = labels.reshape(-1, 5)   # [32*8, 5]
        
        # Create note presence target (1 for note, 0 for no note)
        note_presence_true = (labels_flat[:, 0] != -1).float()
        
        # Note presence loss (binary cross entropy)
        loss_note_presence = F.binary_cross_entropy(
            note_presence_pred.reshape(-1),
            note_presence_true,
            reduction='mean'
        )

        # Time offset loss - only for positions with notes
        valid_notes_mask = (note_presence_true == 1)
        if valid_notes_mask.any():
            loss_time_offset = F.smooth_l1_loss(
                beat_time_offset_pred.reshape(-1)[valid_notes_mask],
                labels_flat[valid_notes_mask, 0]
            )
        else:
            loss_time_offset = torch.tensor(0.0, device=self.device)

        # Only compute other losses where there are actual notes
        if valid_notes_mask.any():
            loss_lineIndex = F.cross_entropy(
                lineIndex_logits.reshape(-1, self.num_classes['lineIndex'])[valid_notes_mask],
                labels_flat[valid_notes_mask, 1].long()
            )
            loss_lineLayer = F.cross_entropy(
                lineLayer_logits.reshape(-1, self.num_classes['lineLayer'])[valid_notes_mask],
                labels_flat[valid_notes_mask, 2].long()
            )
            loss_note_type = F.cross_entropy(
                note_type_logits.reshape(-1, self.num_classes['note_type'])[valid_notes_mask],
                labels_flat[valid_notes_mask, 3].long()
            )
            loss_cut_direction = F.cross_entropy(
                cut_direction_logits.reshape(-1, self.num_classes['cut_direction'])[valid_notes_mask],
                labels_flat[valid_notes_mask, 4].long()
            )
        else:
            loss_lineIndex = loss_lineLayer = loss_note_type = loss_cut_direction = torch.tensor(0.0, device=self.device)

        # Weight and combine losses
        weighted_losses = {
            'note_presence': loss_note_presence * 1.5,  # High weight for note presence
            'time_offset': loss_time_offset * 1.0,
            'note_type': loss_note_type * 1.0,
            'lineIndex': loss_lineIndex * 0.8,
            'lineLayer': loss_lineLayer * 0.8,
            'cut_direction': loss_cut_direction * 0.6
        }

        total_loss = sum(weighted_losses.values())

        # Compute accuracies
        with torch.no_grad():
            note_presence_acc = ((note_presence_pred.reshape(-1) > 0.5) == note_presence_true).float().mean()
            accuracies = {
                'note_presence': note_presence_acc,
                'lineIndex': (lineIndex_logits.reshape(-1, self.num_classes['lineIndex'])[valid_notes_mask].argmax(dim=-1) == labels_flat[valid_notes_mask, 1].long()).float().mean() if valid_notes_mask.any() else torch.tensor(0.0, device=self.device),
                'lineLayer': (lineLayer_logits.reshape(-1, self.num_classes['lineLayer'])[valid_notes_mask].argmax(dim=-1) == labels_flat[valid_notes_mask, 2].long()).float().mean() if valid_notes_mask.any() else torch.tensor(0.0, device=self.device),
                'note_type': (note_type_logits.reshape(-1, self.num_classes['note_type'])[valid_notes_mask].argmax(dim=-1) == labels_flat[valid_notes_mask, 3].long()).float().mean() if valid_notes_mask.any() else torch.tensor(0.0, device=self.device),
                'cut_direction': (cut_direction_logits.reshape(-1, self.num_classes['cut_direction'])[valid_notes_mask].argmax(dim=-1) == labels_flat[valid_notes_mask, 4].long()).float().mean() if valid_notes_mask.any() else torch.tensor(0.0, device=self.device)
            }

        # Logging
        for name, loss in weighted_losses.items():
            self.log(f'train_loss_{name}', loss, on_step=True, on_epoch=True, prog_bar=True)
            if name in accuracies:
                self.log(f'train_acc_{name}', accuracies[name], on_step=True, on_epoch=True, prog_bar=True)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer