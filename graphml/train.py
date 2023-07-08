import lightning as lit
from torch_geometric.loader import DataLoader

import graphml

constants = graphml.Constants()


def main():
    data = graphml.get_graph_data()
    dataset = data.to_torch()
    train_dl = DataLoader([dataset], batch_size=64)
    valid_dl = DataLoader([dataset], batch_size=64)

    model = graphml.GNNModel(
        hidden_channels=128,
        hidden_size=128,
        conv_layers=5,
    )

    trainer = lit.Trainer(
        max_epochs=500,
        accelerator="mps",
        devices=1,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


if __name__ == "__main__":
    main()
