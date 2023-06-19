# Convert Text to Graph

There are many ways of converting a text corpus to a graph. If you have access to named entity recognition (NER) tools that can identify entities in the text, such as drugs and diseases in medical text, genes and proteins in biomedical text, and people, locations, and organizations in news reports, then you can treat these entities as nodes of your graph and draw edges between documents with co-occurring instances of these entities.

Even without a NER tool, you can often use statistical or heuristics-based tools, such as those used to find frequent or likely n-grams, noun phrases, or keywords to identify phrases that can be used as entities and generate graphs from them in a similar manner to entities.

More recently, with the increase in popularity of word embedding approaches, words in documents can be projected from a one-hot encoding in a sparse high dimensional vocabulary space to a dense low dimensional space (generally 80, 100, or 300 dimensions, but we have seen embeddings as high as 768 dimensions) where words with similar neighbors cluster closer together than words that have different neighbors. Such a projection generally results in words with similar meanings clustering together as well.

Somewhat surprisingly, it has been observed that word embeddings can be extended to sentence and document embeddings merely by averaging the embedding vectors for all the words in the sentence or document. Like similar words, sentences and documents that are similar in this space also tend to be similar to each other.

In this milestone, we will use spaCy to generate document embeddings using our paper titles and abstracts, and then compute the similarity between document vectors using cosine similarity. This will result in a dense graph since the distribution of cosine similarity is likely to be fairly continuous between 0 and 1. So we will determine a threshold for similarity based on the distribution and, with all our documents as nodes in a graph, draw edges between any documents where the similarity between them exceeds this threshold.

1. **Load the spaCy language model**. We will represent documents as vectors using word embeddings. The spaCy medium and large provides 300-dimensional word vectors out of the box. Documents are represented by vectors that are the average of their word vectors. In our project, we will use the medium English language model (en_core_web_md).
2. **Extract document vectors**. We loop through the abstracts file provided to us, concatenating the title and text for each abstract. Concatenation involves terminating the title with a period and appending the abstract text to the end, essentially treating the title as the first sentence of the resulting text block representing the abstract, and the abstract as the rest. The concatenated text is passed to the spaCy language model. SpaCy will automatically return the average embedding vector described above for each text block.
3. **Construct a dense document matrix**. The document vectors are composed into a document matrix of size (N, 300), where N represents the number of documents in the collection.
4. **Construct the document similarity matrix**. The document similarity matrix is another NumPy matrix, constructed by computing the dot product of the document matrix and its transpose.
5. **Determine similarity threshold**. The resulting similarity matrix is dense since the document vectors are dense as well. If we built a graph out of this, it would be extremely dense and not provide us with meaningful insights. So we need to establish a similarity threshold above which we will consider a document pair to be related, and below which they would be considered unrelated.
   - We do this by sampling from the similarity matrix and building a histogram of scores. We determine the threshold by inspecting the histogram. The objective is to build an adjacency matrix that represents a graph that is sparse enough to provide a meaningful structure. That is, we connect two documents only if there is a very high similarity.
   - Unfortunately, this is somewhat subjective and you might need a few attempts with different thresholds in order to get a meaningful threshold.
   - Building the histogram against the full document set is quite compute-intensive, so to enable quick development cycles, we sample about 2% of the data (1000 documents) randomly and generate the histogram from it.
   - A good rule of thumb might be to set the threshold to only include edges with weights that are in the top 1–5% of the population.
6. **Create an adjacency matrix**. Remember to set the diagonal to zero. A document is most similar to itself, therefore diagonal elements would be highest, but we are not interested in this relation for our graph.
Update the similarity matrix such that any value above the threshold is 1 and any values below it are 0.
7. **Save the adjacency matrix** . Remember to save the adjacency matrix representing the document graph for the next milestone. We will need the adjacency matrix and a pointer from each row position back to the document ID.

The approach outlined above allows us to leverage document embeddings as features to build an adjacency matrix across the entire corpus.

# Build and Explore a Graph

Workflow

1. **Install Neo4j Server**. Download the latest Neo4j Server from the [Neo4j Download Center](https://neo4j.com/download-center/). 
   - Install Neo4j Server on your disk using [instructions for your specific operating system](https://neo4j.com/docs/operations-manual/current/installation/).
   - Recent versions need the Java Development Kit (JDK) 11.x or newer installed as a prerequisite. Follow the [instructions appropriate to your operating system](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html) to install the JDK.
   - Start the Neo4j server. The directory under which you installed Neo4j will contain a folder with a name starting with `neo4j-community-`. This is your NEO4J_HOME. To start the Neo4j server at your console, run `$NEO4J_HOME/bin/neo4j console` at the command line. You should see the server start up and eventually say that its remote interface is listening on port 7474.
   - Navigate to the Neo4j web interface on your browser—navigate to http://localhost:7474. You will be prompted for the Neo4j user and system password (they are `neo4j` and `neo4j`, respectively). Once you enter them, you will be prompted to change your password. Choose a new password for user neo4j.
   - You should now be able to enter commands to interact with Neo4j. As a test, enter `SHOW DATABASES`; in the browser prompt and hit Control-Enter (or Command-Enter for Mac OS users). You should see the databases neo4j and system.

2. **Configure the Neo4j server**. At this point, our Neo4j server can serve results from user-supplied [Cypher queries](https://neo4j.com/developer/cypher/), but we also want to install the [Graph Data Science (GDS)](https://neo4j.com/docs/graph-data-science/current/) and [Awesome Procedures on Cypher (APOC)](https://neo4j.com/labs/apoc/) plugins.
   - To install the GDS plugin, first stop the Neo4j server (Control-C if you used neo4j console to start or `$NEO4J_HOME/bin/neo4j stop` if you used neo4j start to start the server. 
   - Download the [plugin from the Neo4j Download Center](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-server/) and copy the JAR file to $NEO4J_PLUGINS/plugins/. 
   - To install the [APOC plugin](https://neo4j.com/docs/apoc/current/installation/#apoc), copy the APOC JAR file from $NEO4J_HOME/labs to $NEO4J_HOME/plugins. 
   - Update the Neo4j configuration file at $NEO4J_HOME/conf/neo4j.conf and set the following properties:
     - `apoc.export.file.enabled` to true 
     - `apoc.import.file.enabled` to true
     - `apoc.import.file.use_neo4j_config` to false
     - `dbms.security.procedures.unrestricted` to apoc.*,gds.*
     - `dbms.memory.heap.initial_size` to 8G if you can afford it; otherwise set it to around 50–75% of your available RAM
     - `dbms.memory.heap.max_size` to 8G if you can afford it; otherwise set it to around 50–75% of your available RAM

3. **Create a new database**. This step is not strictly necessary, but in general it is safer to isolate application data from system data, so we will create a separate database to store our graph. That way, if there are issues with the data, we can safely issue a DROP DATABASE command on our database. Unfortunately, Neo4j Community Server doesn’t provide a way to create one from the web browser, but there is a simple hack to do this. In `$NEO4J_HOME/conf/neo4j.conf`, uncomment the property `dbms.default_database` and set it to `av-graph` (that’s the name we give our database).'
   - Restart the Neo4j server. You should see av-graph as a selection option on the left-hand navigation panel of the Neo4j web interface.
4. **Create node.csv and edge.csv files**. We will convert the adjacency matrix we created in the previous milestone into a pair of CSV files nodes.csv and edges.csv. The nodes.csv file contains information about documents—the ID along with metadata such as title and category. The edges.csv file contains information about the links between similar documents in the graph. The header of either file contains important metadata that communicates the graph structure to Neo4j.
5. Create the graph. To create the graph, convert the adjacency matrix we created in the previous milestone into a node list and an edge list. Refer to the instructions under the section Create Input files for neo4j-admin on how to do this. The output of this step should be a pair of files nodes.csv and edges.tsv.
   - Stop the Neo4j server by using the command $NEO4J_HOME/bin/neo4j stop.
   - Run the neo4j-admin command to load the files into Neo4j: $NEO4J_HOME/bin/neo4j-admin import --database=av-graph --nodes=nodes.csv --relationships=edges.csv.
   - Restart Neo4j server using $NEO4J_HOME/bin/neo4j start or $NEO4J_HOME/bin/neo4j console.
6. **Run exploratory commands**. Install the py2neo library using pip install py2neo. We will use this library to connect to Neo4j and issue Cypher queries. Go back to the web interface and examine the graph using some Cypher queries to count the number of nodes and edges in the graph. You can do this either from the Neo4j web interface at http://localhost:7474/ using pure Cypher, or connect to the server from your Jupyter notebook using py2neo.

!!! caution

    Step (2) is outdated. APOC config should be in its own `$NEO4J_HOME/conf/apoc.conf` file.
    Step (5) is outdated. See [import docs](https://neo4j.com/docs/operations-manual/current/tools/neo4j-admin/neo4j-admin-import/)

# Explore a Graph Using Neo4j

