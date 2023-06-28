# Graph Embeddings for Document Similarity

In our first milestone of the project, we will load the graph into Neo4j and generate node2vec node embeddings using the Graph Data Science (GDS) library. These embeddings are dense feature vectors that represent the structure around each node. In this high-dimensional space, nodes that have similar neighborhood structures are closer together than nodes with different neighborhood structures.

Node2vec is one of a family of embeddings that are based on making random walks on graphs. As an NLP practitioner, you are undoubtedly familiar with word2vec. The intuition behind node2vec is similar to word2vec. With word2vec, the idea is to train a network to predict whether a pair of words “belong” together. The belongingness is learned using training data from naturally occurring sentences in text, by finding pairs of words that co-occur within a fixed-size window. With node2vec and its predecessors, random walk-based embedding strategies such as DeepWalk, one constructs sequences of nodes by executing random walks from each node. These sequences are used to find nodes that co-occur within a fixed window size, and a network is trained to predict that these nodes are similar.

Early random walk embedding schemes such as DeepWalk treated each relationship as identical, and the random walks generated were truly random. Node2vec, on the other hand, requires two additional constraints that influence the randomness of the walk. These two parameters are the return factor and the in-out parameter. The return factor determines how often a walk returns to its previously visited node, and the in-out parameter determines whether the walk moves away from the source node or gets closer.

1.  **Load a graph into Neo4j**. Stop the Neo4j server if it is running. Use the neo4j-admin import command to load the provided data files into Neo4j:
   - `$NEO4J_HOME/bin/neo4j-admin import --database=av2-graph --nodes=/path/to/stat_nodes.csv --relationships=/path/to/stat_edges.csv`
   - Restart the server. 
2. **Connect to Neo4j from Python**. We use py2neo as our Python interface and interact with Neo4j from Python over the Jupyter notebook using it. py2neo is a lightweight wrapper which does not require any change in the Cypher query being sent, and thus the queries can be used without change directly on the Neo4j web interface as well. 
3. **Create a virtual graph for GDS**. GDS requires all its algorithms to run within a virtual subgraph. The origin of the requirement seems to have been the assumption that algorithms would need to run on a subset of the data. Hence running algorithms within a subgraph makes sense for performance reasons. In our case, we will run the GDS algorithms against our entire graph. But because of the GDS requirement, we will need to create a virtual subgraph that contains the entire graph. For more on virtual graphs, read the neo4j documentation on creating graphs. 
4. **Generate node2vec vectors**. In node2vec you execute fixed-length random walks from each node in the graph. The intuition is that nodes in each walk will depend on the connectivity between them, hence the walks will incorporate the connectivity information.
- The node2vec family of random walk-based algorithms is influenced to some extent by the word2vec algorithm. You can imagine words in your vocabulary as being a graph, where the edge weight between words w1 and w2 would be determined by the probability that w2 follows w1. These probabilities can be derived by analyzing a large enough corpus of text. With node2vec, we start with a graph and synthesize “sentences” by executing the random walks.
- The analogy described above is almost 100% applicable to a predecessor algorithm to node2vec called DeepWalk. DeepWalk is an example of a first order biased random walk algorithm, meaning that the probability of a node being in a random walk is dependent only on its current state. The node2vec algorithm improves upon DeepWalk by being a second order biased random walk algorithm, meaning that the probability of a node being in a walk is dependent not only on its current state but also on its previous state.
- The second order bias is controlled by two parameters—the returnFactor (p) and inOutFactor (q).
- The returnFactor governs the probability of backtracking and revisiting a previously visited node in the walk. Values lower than 1 encourage backtracking and values higher than 1 discourage it.
- The inOutFactor determines whether the walk will move “outward” or “inward.” Setting inOutFactor to values greater than 1 encourages it to choose nodes closer to the previous node, that is, it encourages breadth-first search (BFS). On the other hand, setting it to values higher than 1 encourages the walk to choose nodes further away from the previous node, that is, it encourages depth-first search (DFS).
- The inOutFactor can be used to tune the behavior of node2vec embeddings to either prefer “homophily” or community-based similarity (for values < 1) or “structural” or connection-based similarity (for values > 1). This is somewhat analogous to syntactic and semantic similarity in NLP.
- Read the node2vec documentation to learn how to call the GDS library. We want to invoke the GDS algorithm to compute the node2vec embeddings and write them back into the database under the node property node2vec_vec key.
- We want to focus on community-based similarity as far as possible, so we use the following parameters for node2vec. You should also experiment with other parameters and observe their effects on the visualizations to get an intuition for how node2vec works.
    ```
    embeddingDimension: 128
    walksPerNode: 10
    walkLength: 80
    windowSize: 3
    inOutFactor: 0.5
    returnFactor: 0.5
    concurrency: 1
    iterations: 25
    writeProperty: ‘node2vec_vec’
    ```


5. **Verify embeddings have been written out**. Verify that the node property node2vec_vec exists and is populated for a random document (say doc_id == "math/0504058"). 
6. **Dump out the content of the database**. Call poc.export.json.all to dump the database to disk in JSON format.

In this liveProject, we start with a new data source, the citation network. This is a set of document nodes and edges, and can therefore be modeled as a graph in Neo4j. In this milestone, we use the node2vec algorithm from the Neo4j Graph Data Science (GDS) to generate node2vec embeddings for each document. These embeddings are based only on the citation connections and thus represent a different approach to the document corpus than our previous embeddings. In addition, while the previous liveProject started from document embeddings and resulted in a graph, this liveProject starts with a graph and results in embeddings.