import os
import typing as T

import dotenv
import neo4j
import prefect

import graphml.constants

dotenv.load_dotenv()

URI = "neo4j://localhost:7687"
AUTH = (os.getenv("DB_USER"), os.getenv("DB_PASS"))
DB = "graphml"

constants = graphml.constants.Constants()


class CategoryEncoder:
    def __init__(self, uri: str, auth: T.Tuple[str, str], db: str):
        self.uri = uri
        self.auth = auth
        self.db = db
        self.data = self.fetch()
        self.id2cat = dict(enumerate(self.data))
        self.cat2id = {v: k for k, v in self.id2cat.items()}

    def fetch(self):
        query = """
        MATCH (a:Article)
        RETURN DISTINCT a.category as category
        """
        data = submit_query(query, self.uri, self.auth, self.db, return_df=True)
        return data["category"].to_list()


def submit_query(
    statement: str, uri: str, auth: T.Tuple[str, str], db: str, return_df: bool = False
):
    with neo4j.GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=db) as session:
            result = session.run(statement)
            if not return_df:
                print(result.data())
            else:
                data = result.to_df(expand=True)
                return data


@prefect.task()
def project_graph(uri: str, auth: T.Tuple[str, str], db: str):
    """Project graph into memory."""
    query = """
    CALL gds.graph.project(
      'articles',
      'Article',
      'CITES'
    ) YIELD
      graphName AS graph, nodeProjection, nodeCount AS nodes, relationshipProjection,
      relationshipCount AS rels
    """
    submit_query(query, uri, auth, db, return_df=False)


@prefect.task()
def run_node2vec(uri: str, auth: T.Tuple[str, str], db: str):
    """Run Node2Vec algorithm and write embeddings back to database."""
    query = """
    CALL gds.beta.node2vec.write(
        'articles',
        {
            writeProperty: 'node2vec',
            embeddingDimension: 128,
            windowSize: 3,
            inOutFactor: 0.5,
            returnFactor: 0.5,
            iterations: 25
        }
    )
    YIELD nodeCount, computeMillis, configuration
    """
    submit_query(query, uri, auth, db, return_df=False)


@prefect.task()
def dump_db(filename: str, uri: str, auth: T.Tuple[str, str], db: str):
    """Dump database objects to CSV file."""
    query = """
    MATCH(node)
    RETURN node.docId as docId, node.title as title, node.category as category,
    node.node2vec as embed
    """
    data = submit_query(query, uri, auth, db, return_df=True)
    filepath = constants.DATA.joinpath(filename)
    data.to_csv(filepath, index=False, compression="gzip")


@prefect.flow(name="Embed paper citations via Node2Vec")
def main():
    """Embedding workflow."""
    project_graph(URI, AUTH, DB)
    run_node2vec(URI, AUTH, DB)
    dump_db(filename="embed_cites.csv.gz", uri=URI, auth=AUTH, db=DB)


if __name__ == "__main__":
    main()
