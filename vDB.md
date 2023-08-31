# Top 10 Best Vector Databases & Libraries

Elasticsearch (64.9k ⭐) 
  — A distributed search and analytics engine that supports various types of data. 
  - One of the data types that Elasticsearch supports is vector fields, which store dense vectors of numeric values.
  - In version 7.10, Elasticsearch added support for indexing vectors into a specialized data structure to support fast kNN retrieval through the kNN search API.
  - In version 8.0, Elasticsearch added support for native natural language processing (NLP) with vector fields.

Faiss (24.1k ⭐) 
  — A library for efficient similarity search and clustering of dense vectors. 
  - It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
  - It also contains supporting code for evaluation and parameter tuning.
  - It is developed primarily at Meta’s Fundamental AI Research group.

Milvus (22.4k ⭐) 
  — An open-source vector database that can manage trillions of vector datasets and supports multiple vector search indexes and built-in filtering.

Qdrant (12.5k ⭐) 
  — A vector similarity search engine and vector database. 
  - It provides a production-ready service with a convenient API to store, search, and manage points vectors with an additional payload.
  - Qdrant is tailored to extended filtering support.
  - It makes it useful for all sorts of neural-network or semantic-based matching, faceted search, and other applications.

Chroma (8.2k ⭐) 
  — An AI-native open-source embedding database. 
  - It is simple, feature-rich, and integrable with various tools and platforms for working with embeddings.
  - It also provides a JavaScript client and a Python API for interacting with the database.

OpenSearch (7.4k ⭐) 
  — A community-driven, open source fork of Elasticsearch and Kibana following the license change in early 2021. 
  - It includes a vector database functionality that allows you to store and index vectors and metadata, and perform vector similarity search using k-NN indexes.

Weaviate (7.3k ⭐) 
  — An open-source vector database that allows you to store data objects and vector embeddings from your favorite ML-models, and scale seamlessly into billions of data objects.

Vespa(4.6k ⭐) 
  — A fully featured search engine and vector database. 
  - It supports vector search (ANN), lexical search, and search in structured data, all in the same query.
  - Integrated machine-learned model inference allows you to apply AI to make sense of your data in real time.

pgvector (5.3k ⭐) 
  — An open-source extension for PostgreSQL that allows you to store and query vector embeddings within your database. 
  - It is built on top of the Faiss library, which is a popular library for efficient similarity search of dense vectors.
  - pgvector is easy to use and can be installed with a single command.

Vald (1.3k ⭐) 
  — A highly scalable distributed fast approximate nearest neighbor dense vector search engine. 
  - Vald is designed and implemented based on the Cloud-Native architecture.
  - It uses the fastest ANN Algorithm NGT to search neighbors.
  - Vald has automatic vector indexing and index backup, and horizontal scaling which made for searching from billions of feature vector data.
