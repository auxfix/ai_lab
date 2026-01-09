import hashlib
import json

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class CodeVectorizer:
    """Turns code into magical numbers (embeddings)"""

    def __init__(self, model_name="all-MiniLM-L6-v2", persist_dir="./chroma_db", use_gpu=True):
        print(f"🚀 Loading embedding model: {model_name}")
        
        # Use GPU if available and requested
        device = "cuda" if use_gpu else "cpu"
        self.embedding_model = SentenceTransformer(model_name, device=device)
        
        if use_gpu and self.embedding_model.device.type == "cuda":
            print(f"⚡ GPU acceleration enabled: {self.embedding_model.device}")
        else:
            print(f"🐌 Running on CPU")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="codebase_rag",
            metadata={"hnsw:space": "cosine"},  # Cosine similarity works well for code
        )

        print(f"📊 Collection ready: {self.collection.name}")

    def generate_id(self, chunk):
        """Create a unique ID for each chunk (deterministic)"""
        content_hash = hashlib.md5(chunk["content"].encode()).hexdigest()
        return f"{chunk['source_file']}_{chunk['chunk_id']}_{content_hash[:8]}"

    def embed_and_store(self, chunks, batch_size=256):  # Larger batches with your GPU
        """Embed chunks and store in vector database"""
        if not chunks:
            print("⚠️ No chunks to embed!")
            return

        print(f"🧠 Embedding {len(chunks)} chunks...")

        # Process in batches to avoid memory issues
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            batch_embeddings = []

            for chunk in batch:
                # Generate embedding
                embedding = self.embedding_model.encode(chunk["content"]).tolist()

                # Prepare metadata
                metadata = {
                    "source_file": chunk["source_file"],
                    "language": chunk["language"],
                    "chunk_id": chunk["chunk_id"],
                    "total_chunks": chunk["total_chunks"],
                    "full_path": chunk["full_path"],
                }

                batch_ids.append(self.generate_id(chunk))
                batch_documents.append(chunk["content"])
                batch_metadatas.append(metadata)
                batch_embeddings.append(embedding)

            # Store batch
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings,
            )

            print(
                f"💾 Saved batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}"
            )

        print(f"✅ Stored {len(chunks)} chunks in vector database")
        print(f"📈 Collection count: {self.collection.count()}")

    def search_similar(self, query, n_results=5):
        """Search for similar code chunks"""
        # Embed the query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def get_stats(self):
        """Get database statistics"""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }


# Test embedding
if __name__ == "__main__":
    vectorizer = CodeVectorizer()

    # Test search with dummy data
    test_chunks = [
        {
            "content": "def get_user(id):\n    return db.query(User).filter(User.id == id).first()",
            "source_file": "test.py",
            "language": "py",
            "chunk_id": 0,
            "total_chunks": 1,
            "full_path": "test.py",
        }
    ]

    vectorizer.embed_and_store(test_chunks)

    # Test search
    results = vectorizer.search_similar("How to fetch user from database?")
    print("\n🔍 Search results:")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"\n{i + 1}. Similarity: {1 - dist:.3f}")
        print(f"   File: {meta['source_file']}")
        print(f"   Preview: {doc[:100]}...")
