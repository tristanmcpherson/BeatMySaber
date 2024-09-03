import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiTransformerModel(pl.LightningModule):
    def __init__(self, mfcc_dim, chroma_dim, contrast_dim, output_dim, nhead, nhid, nlayers):
        super(MultiTransformerModel, self).__init__()
        
        # Define separate Transformers for each feature set
        self.mfcc_transformer = nn.Transformer(mfcc_dim, nhead, nhid, nlayers)
        self.chroma_transformer = nn.Transformer(chroma_dim, nhead, nhid, nlayers)
        self.contrast_transformer = nn.Transformer(contrast_dim, nhead, nhid, nlayers)
        
        # Final fully connected layer
        combined_dim = mfcc_dim + chroma_dim + contrast_dim
        self.fc = nn.Linear(combined_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src):
        # Split input features into their respective groups
        mfcc_features = src[:, :, :13]  # Assuming first 13 dims are MFCC
        chroma_features = src[:, :, 13:25]  # Next 12 dims are Chroma
        contrast_features = src[:, :, 25:]  # Remaining dims are Spectral Contrast
        
        # Pass each group through its respective Transformer
        mfcc_output = self.mfcc_transformer(mfcc_features.permute(1, 0, 2))
        chroma_output = self.chroma_transformer(chroma_features.permute(1, 0, 2))
        contrast_output = self.contrast_transformer(contrast_features.permute(1, 0, 2))
        
        # Concatenate the outputs along the feature dimension
        combined_output = torch.cat((mfcc_output, chroma_output, contrast_output), dim=2)
        
        # Pass the combined output through the final fully connected layer
        output = self.fc(combined_output.permute(1, 0, 2))
        return output

    def training_step(self, batch, batch_idx):
        features, labels = batch
        output = self(features)
        loss = self.loss_fn(output.view(-1, output.shape[-1]), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
