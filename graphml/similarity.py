import dpipes
import numpy as np

import graphml

constants = graphml.Constants()


def load_embed_matrix():
    filepath = constants.DATA.joinpath("embed_matrix.csv.gz")
    return np.loadtxt(fname=str(filepath), delimiter=",")


def get_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Get a similarity matrix."""
    return np.corrcoef(matrix)


def get_adjacency_matrix(matrix: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    """Get an adjacency matrix for nodes exceeding threshold."""
    adj = np.where(matrix > threshold, 1.0, 0.0)
    mask = 1 - np.eye(len(adj))
    return adj * mask


def save_matrix(matrix: np.ndarray):
    """Save matrix to CSV file."""
    filepath = constants.DATA.joinpath("adj_matrix.csv.gz")
    np.savetxt(fname=str(filepath), X=matrix, delimiter=",")


processor = dpipes.Pipeline(funcs=[get_similarity_matrix, get_adjacency_matrix, save_matrix])


if __name__ == "__main__":
    data = load_embed_matrix()
    processor(data)
