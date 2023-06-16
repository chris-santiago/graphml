import typing as T

import dpipes
import numpy as np
import pandas as pd
import spacy
import tqdm

import graphml

constants = graphml.Constants()


def load_data() -> pd.DataFrame:
    """Load stat abstracts data."""
    file = constants.DATA.joinpath("stat-abstracts.tsv")
    return pd.read_csv(file, sep="\t", header=None)


def concat_titles_abstracts(data: pd.DataFrame) -> pd.Series:
    """Concatenate paper titles and abstracts into single string."""
    return data.iloc[:, 1] + "." + data.iloc[:, 3]


def process_docs(
    docs: pd.Series, model: str = "en_core_web_md", batch_size: int = 2000, n_process: int = 5
) -> T.Iterator:
    """Process a series a documents using a spaCy pipeline."""
    nlp = spacy.load(model)
    return nlp.pipe(docs, batch_size=batch_size, n_process=n_process)


def get_embedding_matrix(docs: T.Iterator) -> np.ndarray:
    """Get document embedding matrix."""
    total = len(docs.gi_frame.f_locals["texts"])
    return np.array([d.vector for d in tqdm.tqdm(docs, total=total)])


def save_matrix(matrix: np.ndarray):
    """Save matrix to CSV file."""
    filepath = constants.DATA.joinpath("embed_matrix.csv.gz")
    np.savetxt(fname=str(filepath), X=matrix, delimiter=",")


preprocessor = dpipes.Pipeline(
    funcs=[concat_titles_abstracts, process_docs, get_embedding_matrix, save_matrix]
)


if __name__ == "__main__":
    data = load_data()
    preprocessor(data)
