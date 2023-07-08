import lightning as lit
from torch_geometric.loader import DataLoader

import graphml

constants = graphml.Constants()


class MLPBase(graphml.GNNModel):
    def forward(self, x, edge_idx):
        return self.head(x)


def main():
    data = graphml.get_graph_data()
    dataset = data.to_torch()
    train_dl = DataLoader([dataset], batch_size=64)
    valid_dl = DataLoader([dataset], batch_size=64)

    model = MLPBase(
        hidden_channels=dataset.x.shape[1],  # ensure input size matches w/o GCN layers
        hidden_size=128,
    )

    trainer = lit.Trainer(
        max_epochs=500,
        accelerator="mps",
        devices=1,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


if __name__ == "__main__":
    main()
