# Graph Embeddings for Document Similarity

## Construct Node2Vec Embeddings

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

## Visualize Citation Embeddings

1. **Extract document IDs, vectors, and labels.**  The neo4j-dump.json file that we extracted in the previous milestone of this liveProject contains all the information about the Neo4j database. The file is in JSON-L format, which means that each node and relationship record is on its own line as a well-formed JSON structure. The data we need for our current milestone is the document ID, the document vector generated by node2vec, and the category label. They can be found in the JSON at the following locations.
   - doc_id: data['properties']['doc_id']
   - vector: data['properties']['node2vec_vec']
   - label: data['properties']['category']
   - Extract these variables for each node into rhw Python list variables docids, vectors, and labels.
   - At the end of this step each list should have 50426 elements in it.
   - When adding the vector to the list vectors, convert it to a NumPy array first by wrapping the list vector in a np.array(vector). That way the vectors list is a list of numpy.array objects.
2. **Use UMAP for dimensionality reduction**.  Convert the list vectors of numpy.array objects to a matrix. This is done by wrapping the list in another np.array(vectors).
   - Check that you now have a matrix of size (50426, 128), since each vector has 128 elements.
   N- ext convert the labels to a set of numeric IDs. This can be done by creating a dictionary of category strings to label id and then looking up the ID from the label to create a new variable label_ids.
   - Create a UMAP mapper. It is worth experimenting with a few hyperparameters for mapper construction based on information in the UMAP parameters page and seeing what the resulting visualization is like and whether it aligns with what we believe the data should look like.
   - In our case, we will create the UMAP mapper with the following parameters:
     - n_neighbors 10
     - min_dist 0.1
     - metric “cosine”
   - Although we are using UMAP here to reduce the dimension of X to 2D, UMAP allows reducing to other dimensions as well.
   - We then plot the points in 2D using umap.plot.points. The plot indicates two major clusters. Each point represents the node2vec embedding for a document projected by UMAP onto 2D space. Each point has an associated color that indicates its category. Documents of all categories seem to be evenly distributed in both clusters.
   - So it looks like node2vec has discovered a coarser structure in the graph that was not implied by the category labels. This makes intuitive sense since citations reflect acknowledgments of past work, and papers in one area of study are very likely to be dependent on papers in other areas of study.
3. **Use clustering for HDBSCAN**.  Rather than depend only on visual identification of clusters, we can use a clustering algorithm such as HDBSCAN to discover clusters for us. We will cluster on reduced 2D space generated by UMAP rather than the full 128-dimensional node2vec vector space. This is because it is generally very difficult to find clusters in high dimensions because high dimensional data is sparse. As with UMAP, it is worth trying out different hyperparameter values for HDBSCAN as well. This page describes the various available HDBSCAN hyperparameters. We will use the following HDBSCAN parameters to train the clusterer.
   - min_cluster_size 40
   - To keep the output to not be too dense, we sample 5000 points from our input and plot those as a 2D scatterplot. As with UMAP, HDBSCAN also predicts two major clusters. 
4. **Examine the composition of the largest clusters**.  We have seen that UMAP dimensionality reduction and HDBSCAN clustering both show two major clusters in the data. We also notice that the categories seem to be distributed uniformly across both clusters and that the node2vec embedding does not seem to capture this information.
- We next examine the composition (in terms of document categories) of documents in each of the two largest clusters. First, we find the two largest clusters by counting the labels returned by clusterer.labels_.
- Remember that HDBSCAN, like other density-based clusterers, will assign an “unknown” or “other” label to documents it is unable to cluster. The "-1" cluster can be quite large and have no discernible structure, so we want to ignore it when we find the two largest clusters.
- For each of the two clusters, we count the different number of labels and plot a bar chart of label counts (y-axis) versus labels (x-axis).
- We observe that stat.ML (machine learning) is dominant in both clusters. However, in one cluster other subject areas are more common compared to stat.ML, implying that this cluster may have a more “rounded” set of articles than the other. In both cases, machine learning papers seem to be very popular in the field of statistics.
