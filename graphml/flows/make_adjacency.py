import numpy as np
import polars as pl
import prefect
import scipy.sparse as sparse

import graphml

constants = graphml.Constants()
SIMILARITY_THRESHOLD = 0.985


@prefect.task()
def load_embed_matrix() -> pl.DataFrame:
    """Load embedding matrix."""
    filepath = constants.DATA.joinpath("embed_matrix.csv.gz")
    return pl.read_csv(filepath, has_header=False)


@prefect.task()
def get_similarity_matrix(matrix: pl.DataFrame) -> pl.DataFrame:
    """Get a similarity matrix."""
    # polars implementation calls .transpose prior to np.corrcoef
    # https://github.com/pola-rs/polars/issues/9430
    return matrix.transpose().corr()


@prefect.task()
def get_adjacency_matrix(matrix: pl.DataFrame, threshold: float = 0.98) -> sparse.csr_matrix:
    """Get a sparse adjacency matrix for nodes exceeding threshold."""
    adj = np.where(matrix.to_numpy() > threshold, 1, 0).astype(np.int8)
    mask = 1 - np.eye(len(adj))
    return sparse.csr_matrix(adj * mask.astype(np.int8))


@prefect.task()
def save_sparse_matrix(matrix: sparse.csr_matrix):
    """Save matrix to CSV file."""
    filepath = constants.DATA.joinpath("adj_matrix.npz")
    sparse.save_npz(file=str(filepath), matrix=matrix, compressed=True)


@prefect.flow(name="Create adjacency matrix from embeddings.")
def main():
    """Adjacency matrix flow."""
    embed = load_embed_matrix()
    sim = get_similarity_matrix(embed)
    adj = get_adjacency_matrix(sim, threshold=SIMILARITY_THRESHOLD)
    save_sparse_matrix(adj)


if __name__ == "__main__":
    main()
