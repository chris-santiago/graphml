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

# Explore a Graph Using Neo4j

