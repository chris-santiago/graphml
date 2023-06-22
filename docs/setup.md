# Setup

## Environment

- `pdm`
- `Taskfile`

*Libraries*:

- `matplotlib`
- `networkx`
- `numpy`
- `pandas`
- `spacy`
- `py2neo`
- `umap`
- `hdbscan`

We'll use the medium English language model with spaCy. Download instructions [here](https://spacy.io/usage) or
run `task install-elm` if you have `Taskfile` installed on your system.

## Neo4j Database

Download the Neo4j server (Community Edition) from the [Neo4j Download Center](https://neo4j.com/download-center/) and install it locally using [instructions](https://neo4j.com/docs/operations-manual/current/installation/osx/) for your specific operating system. You will need the Java Development Kit (JDK) 11.x or newer. If you don’t have it already, you will need to download and install it as well, as a prerequisite.

### Configure the Neo4j server

Update the $NEO4J_HOME/config/neo4j.cfg file and set the following properties:

```
apoc.export.file.enabled=true
apoc.import.file.use_neo4j_config=false
apoc.import.file.enabled=true
dbms.security.procedures.unrestricted=apoc.*,gds.*
# if you can afford it; otherwise set it to around 50–75% of your available RAM
dbms.memory.heap.initial_size=8G
# if you can afford it; otherwise set it to around 50–75% of your available RAM
dbms.memory.heap.max_size=8G

```

!!! note

    Those configs look like they may be for an older version of Neo4j.
    Py2Neo is EOL; use official Neo4j drivers!

## Data

