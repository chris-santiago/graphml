import polars as pl
import prefect
import scipy.sparse as sparse

import graphml

constants = graphml.Constants()


@prefect.task()
def load_adjacency_matrix():
    """Load a CSR adjacency matrix."""
    filepath = constants.DATA.joinpath("adj_matrix.npz")
    return sparse.load_npz(str(filepath)).tocoo()


@prefect.task()
def load_nodes():
    """Load node data."""
    filepath = constants.DATA.joinpath("nodes.csv")
    return pl.read_csv(
        filepath,
        has_header=False,
        infer_schema_length=0,
        new_columns=["docId", "title", "category", "label"],
    )


@prefect.task()
def make_edges(matrix: sparse.coo_matrix, nodes: pl.DataFrame) -> pl.DataFrame:
    """
    Make edges between similar nodes.

    Parameters
    ----------
    matrix: coo_matrix
        A sparse COO matrix representing node adjacency.
    nodes: pl.DataFrame
        DataFrame containing node IDs and properties.

    Returns
    -------
    pl.DataFrame
        A DataFrame of node sources, targets and relationship type.
    """
    edges = pl.DataFrame({"source": matrix.row, "target": matrix.col})
    edges = edges.filter(pl.col("source") < pl.col("target"))  # remove symmetric edges
    nodes2ids = dict(enumerate(nodes["docId"]))
    return edges.with_columns(
        pl.col(["source", "target"]).map_dict(nodes2ids), pl.lit("similarTo").alias("type")
    )


@prefect.task()
def save_edges(edges: pl.DataFrame):
    """Save edges data to CSV file."""
    fp = constants.DATA.joinpath("edges.csv")
    edges.write_csv(fp, has_header=False)


@prefect.flow(name="Create graph edges from adjacency matrix.")
def main():
    """Edge workflow."""
    adj = load_adjacency_matrix()
    nodes = load_nodes()
    edges = make_edges(adj, nodes)
    save_edges(edges)


if __name__ == "__main__":
    main()
