import typing as T

import lightning as lit
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torchmetrics


class GCNConvBlock(nn.Module):
    def __init__(self, hidden_channels: int, p_drop: float):
        super().__init__()
        self.block = gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GCNConv(in_channels=-1, out_channels=hidden_channels), "x, edge_index -> x"),
                (nn.ReLU(inplace=True), "x -> x"),
                nn.Dropout(p=p_drop),
            ],
        )

    def forward(self, x, edge_idx):
        return self.block(x, edge_idx)


class GNNModel(lit.LightningModule):
    def __init__(
        self,
        hidden_channels: int = 32,
        conv_layers: int = 3,
        hidden_size: int = 64,
        p_drop: float = 0.2,
        n_classes: int = 6,
        optim: T.Optional[T.Callable[[torch.optim.Optimizer], torch.optim.Optimizer]] = None,
        score_func: T.Optional[torchmetrics.Metric] = None,
    ):
        super().__init__()
        self.optim = optim
        self.loss_func = nn.NLLLoss()
        self.score_func = (
            score_func
            if score_func
            else torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        )
        self.save_hyperparameters()

        self.gcn_layers = nn.ModuleList(
            [
                GCNConvBlock(hidden_channels=hidden_channels, p_drop=p_drop)
                for _ in range(conv_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=hidden_channels, out_features=hidden_size),
            nn.Dropout(p=p_drop),
            nn.Linear(in_features=hidden_size, out_features=n_classes),
            nn.LogSoftmax(),
        )

    def forward(self, x, edge_idx):
        for layer in self.gcn_layers:
            embeds = layer(x, edge_idx)
        return self.head(embeds)

    def get_results(self, batch):
        x, edge_idx, y = batch.x, batch.edge_index, batch.y
        labels = self(x, edge_idx)
        loss = self.loss_func(labels, y)
        preds = torch.argmax(labels, dim=1)
        return loss, preds

    def training_step(self, batch, idx):
        loss, _ = self.get_results(batch)
        self.log("train-loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, idx):
        loss, preds = self.get_results(batch)
        self.score_func.update(preds, batch.y)
        self.log("valid-loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "valid-acc", self.score_func, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, idx):
        loss, preds = self.get_results(batch)
        self.score_func.update(preds, batch.y)
        self.log("test-loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "test-acc", self.score_func, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        if self.optim:
            return self.optim(self.parameters())
        return torch.optim.Adam(self.parameters())
