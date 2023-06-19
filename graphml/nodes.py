import polars as pl
import scipy.sparse as sparse

import graphml

constants = graphml.Constants()


def make_nodes() -> pl.DataFrame:
    filepath = constants.DATA.joinpath("stat-abstracts.tsv")
    # Weird bug where only half of rows are read if specifying columns in `read_csv` call
    abstracts = pl.read_csv(
        filepath,
        has_header=False,
        separator="\t",
        infer_schema_length=0,
    )
    return abstracts.select(pl.col("column_1").alias("ID"), pl.col("column_2").alias("TITLE"))


def make_edges() -> pl.DataFrame:
    filepath = constants.DATA.joinpath("adj_matrix.npz")
    adj = sparse.load_npz(str(filepath)).tocoo()
    return pl.DataFrame({"SOURCE": adj.row, "TARGET": adj.col})


def map_ids_to_edges(nodes: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    nodes2ids = dict(enumerate(nodes["ID"]))
    return edges.select(pl.all().map_dict(nodes2ids))


if __name__ == "__main__":
    nodes = make_nodes()
    edges = make_edges()
    edges = map_ids_to_edges(nodes, edges)

    fp = constants.DATA.joinpath("nodes.csv")
    nodes.write_csv(fp)

    fp = constants.DATA.joinpath("edges.csv")
    edges.write_csv(fp)
