import polars as pl
import prefect

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
def make_nodes(data: pl.DataFrame) -> pl.DataFrame:
    """Extract paper ID, title and category to create nodes."""
    return data.select(
        pl.col("docId"),
        pl.col("title"),
        pl.col("category").str.split(";").list.first(),  # Grab first category
        pl.lit("Paper").alias("label"),
    )


@prefect.task()
def save_nodes(nodes: pl.DataFrame):
    """Save nodes to CSV file."""
    fp = constants.DATA.joinpath("nodes.csv")
    nodes.write_csv(fp, has_header=False)


@prefect.flow(name="Create graph nodes from text data.")
def main():
    """Node workflow."""
    data = load_abstract_data()
    nodes = make_nodes(data)
    save_nodes(nodes)


if __name__ == "__main__":
    main()
