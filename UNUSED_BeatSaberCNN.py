import torch.nn as nn
import torch.nn.functional as F
import torch


class BeatSaberCNN(nn.Module):
    def __init__(self, n_features, time_steps):
        super(BeatSaberCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Compute flattened size dynamically
        self.flattened_size = self._get_flattened_size(n_features, time_steps)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        # Output layers
        self.fc_time_offset = nn.Linear(128, 1)
        self.fc_lineIndex = nn.Linear(128, 5)        # Classes:  
        self.fc_lineLayer = nn.Linear(128, 4)        # Classes:
        self.fc_note_type = nn.Linear(128, 4)        # Classes: 0, 1, 2 ("no note")
        self.fc_cut_direction = nn.Linear(128, 10)   # Classes: 0-8, 9 ("no note")

    def _get_flattened_size(self, n_features, time_steps):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_features, time_steps)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        beat_time_offset = self.fc_time_offset(x).squeeze(1)

        lineIndex_logits = self.fc_lineIndex(x)
        lineLayer_logits = self.fc_lineLayer(x)
        note_type_logits = self.fc_note_type(x)
        cut_direction_logits = self.fc_cut_direction(x)
        return beat_time_offset, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits
