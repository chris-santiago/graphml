import typing as T

import polars as pl
import prefect
import spacy
import tqdm

import graphml

constants = graphml.Constants()


@prefect.task()
def load_abstract_data():
    """Load paper abstract data."""
    file = constants.DATA.joinpath("stat-abstracts.tsv")
    return pl.read_csv(
        file,
        separator="\t",
        infer_schema_length=0,
        has_header=False,
        new_columns=["docId", "title", "category", "abstract"],
    )


@prefect.task()
def concat_titles_abstracts(data: pl.DataFrame) -> pl.Series:
    """Concatenate titles and abstracts."""
    return data.select(pl.col("title") + pl.lit(".") + pl.col("abstract")).to_series()


@prefect.task()
def process_docs(
    docs: pl.Series, model: str = "en_core_web_md", batch_size: int = 2000, n_process: int = 5
) -> T.Iterator:
    """
    Process a series a documents using a spaCy pipeline.

    Parameters
    ----------
    docs: pl.Series
        A Series of documents to process.
    model: str
        Name of SpaCy NLP model to use.
    batch_size: int
        Batch size for SpaCy NLP pipeline.
    n_process: int
        Number of CPU processes for SpaCy NLP pipeline.

    Returns
    -------
    Iterator
        An Iterator(Generator) for SpaCy NLP pipeline.
    """
    nlp = spacy.load(model)
    return nlp.pipe(docs.to_list(), batch_size=batch_size, n_process=n_process)


@prefect.task()
def get_embedding_matrix(docs: T.Iterator, n_docs: T.Optional[int] = None) -> pl.DataFrame:
    """
    Get document embedding matrix.

    Parameters
    ----------
    docs: Iterator
        An Iterator(Generator) for SpaCy NLP pipeline.
    n_docs: Optional[int]
        Number of total docs, if known; for use with TQDM counter.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing document embeddings.
    """
    if n_docs:
        return pl.DataFrame([d.vector for d in tqdm.tqdm(docs, total=n_docs)], orient="row")
    return pl.DataFrame([d.vector for d in docs], orient="row")


@prefect.task()
def save_embedding_matrix(matrix: pl.DataFrame):
    """Save matrix to CSV file."""
    filepath = constants.DATA.joinpath("embed_matrix.csv.gz")
    matrix.write_csv(filepath, has_header=False)


@prefect.flow(name="Preprocess text data into embedding matrix.")
def main():
    """Doc embedding flow."""
    data = load_abstract_data()
    docs = concat_titles_abstracts(data)
    docs = process_docs(docs)
    embed = get_embedding_matrix(docs, n_docs=len(data))
    save_embedding_matrix(embed)


if __name__ == "__main__":
    main()
