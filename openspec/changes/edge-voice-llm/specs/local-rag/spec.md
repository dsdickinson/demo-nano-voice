## ADDED Requirements

### Requirement: Document Ingestion and Chunking
The system SHALL ingest text and markdown documents from a configured local directory (`knowledge/`) and split them into overlapping context chunks with metadata.

#### Scenario: Ingesting text files from folder
- **WHEN** text or markdown files are placed in the knowledge directory
- **THEN** the ingestion module SHALL parse the documents into text chunks of configurable token length and overlap

### Requirement: Vector Indexing and Semantic Similarity Search
The system SHALL embed document chunks using a local micro-embedding model and store them in an in-memory vector index (such as FAISS) for fast cosine similarity search.

#### Scenario: Querying relevant context
- **WHEN** a natural language user query is submitted to the vector retriever
- **THEN** the retriever SHALL return the top-k most semantically relevant text chunks within 30 milliseconds

### Requirement: Prompt Context Augmentation
The system SHALL format retrieved knowledge chunks into the system prompt prefix for the downstream LLM generation engine.

#### Scenario: Prompt synthesis with retrieved context
- **WHEN** relevant knowledge chunks are retrieved for a query
- **THEN** the query handler SHALL wrap the context and user query into a structured prompt conforming to the assistant's instruction format

### Requirement: Profile-Scoped Knowledge Isolation
The system SHALL isolate knowledge bases and vector indices according to the selected active profile (such as `public` or `private`).

#### Scenario: Querying with public profile
- **WHEN** the system is running with the public profile
- **THEN** vector retrieval SHALL only search documents within the public knowledge directory and never access private knowledge documents

#### Scenario: Querying with private profile
- **WHEN** the system is running with the private profile
- **THEN** vector retrieval SHALL index and search private knowledge documents without modifying or exposing them to the public index
