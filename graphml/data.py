import dataclasses
import typing as T

import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.transforms import RandomNodeSplit

import graphml

constants = graphml.Constants()


class DocIdEncoder:
    def __init__(self, ids: T.Iterable[str]):
        self.idx2doc = dict(enumerate(ids))
        self.doc2idx = {v: k for k, v in self.idx2doc.items()}


class CategoryEncoder:
    def __init__(self, categories: T.Iterable[str]):
        self.id2cat = dict(enumerate(categories))
        self.cat2id = {v: k for k, v in self.id2cat.items()}


@dataclasses.dataclass
class GraphData:
    vectors: pl.DataFrame
    nodes: pl.DataFrame
    edges: pl.DataFrame

    def __post_init__(self):
        self.doc_enc = DocIdEncoder(self.nodes.get_column("docId"))
        self.cat_enc = CategoryEncoder(self.nodes.get_column("category").unique())

    def to_torch(self):
        mapped_edges = self.edges.with_columns(
            pl.col("source").map_dict(self.doc_enc.doc2idx),
            pl.col("target").map_dict(self.doc_enc.doc2idx),
        )

        data = Data(
            x=torch.tensor(self.vectors.to_numpy(), dtype=torch.float32),
            edge_index=torch.tensor(mapped_edges.transpose().to_numpy(), dtype=torch.int32),
            y=torch.tensor(
                self.nodes.get_column("category").map_dict(self.cat_enc.cat2id).to_numpy(),
                dtype=torch.int8,
            ),
        )
        data.validate()

        splitter = RandomNodeSplit()
        return LightningNodeData(splitter(data), loader="full")


def get_graph_data():
    vectors = pl.read_csv(
        constants.DATA.joinpath("stat-abstract-vectors.csv"),
        has_header=False,
    ).drop("column_1")

    edges = pl.read_csv(
        constants.DATA.joinpath("stat-edges.csv"),
        has_header=True,
        columns=[0, 1],
        new_columns=["source", "target"],
    )

    nodes = pl.read_csv(
        constants.DATA.joinpath("stat-nodes.csv"),
        has_header=True,
        columns=[0, 1, 2],
        new_columns=["docId", "title", "category"],
        infer_schema_length=None,
    )

    return GraphData(vectors, nodes, edges)
