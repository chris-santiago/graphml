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
    return abstracts.select(
        pl.col("column_1").alias("ID"),
        pl.col("column_2").alias("TITLE"),
        pl.col("column_3").str.extract(r"(\w+)", 1).alias("CATEGORY"),
    )


def make_edges() -> pl.DataFrame:
    filepath = constants.DATA.joinpath("adj_matrix.npz")
    adj = sparse.load_npz(str(filepath)).tocoo()
    return pl.DataFrame({"SOURCE": adj.row, "TARGET": adj.col, "TYPE": "SIMILAR_TO"})


def map_ids_to_edges(nodes: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    nodes2ids = dict(enumerate(nodes["ID"]))
    return edges.select(pl.col(["SOURCE", "TARGET"]).map_dict(nodes2ids))


if __name__ == "__main__":
    nodes = make_nodes()
    nodes.with_columns(pl.lit("StatsPaper").alias("LABEL"))
    edges = make_edges()
    edges = map_ids_to_edges(nodes, edges)
    edges = edges.with_columns(pl.lit("SIMILAR_TO").alias("TYPE"))

    fp = constants.DATA.joinpath("nodes.csv")
    nodes.write_csv(fp, has_header=False)

    fp = constants.DATA.joinpath("edges.csv")
    edges.write_csv(fp, has_header=False)
