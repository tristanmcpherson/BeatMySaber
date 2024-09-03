class BeatSaberModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, nhead, nhid, nlayers):
        super(BeatSaberModel, self).__init__()
        self.transformer = nn.Transformer(input_dim, nhead, nhid, nlayers)
        self.fc = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src):
        # Transformer expects (sequence_length, batch_size, input_dim)
        src = src.permute(1, 0, 2)  # Reorder for transformer
        output = self.transformer(src)
        output = self.fc(output)
        return output.permute(1, 0, 2)  # Reorder back for output

    def training_step(self, batch, batch_idx):
        features, labels = batch
        output = self(features)
        loss = self.loss_fn(output.view(-1, output.shape[-1]), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer