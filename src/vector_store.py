"""
Vector Store Loader and Manager

This module handles loading and managing the vector store for the RAG system.
It supports loading pre-built embeddings from parquet files or building
a vector store from filtered complaints data.
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional, Tuple


class VectorStoreManager:
    """Manages vector store operations for complaint retrieval."""
    
    def __init__(
        self,
        persist_directory: str = "vector_store/chroma",
        collection_name: str = "complaint_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model_name: Name of the sentence transformer model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB client
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def load_prebuilt_embeddings(self, parquet_path: str) -> bool:
        """
        Load pre-built embeddings from a parquet file.
        
        The parquet file should have columns:
        - id: string
        - document: string (text chunk)
        - embedding: list of doubles
        - metadata: struct with fields (chunk_index, company, complaint_id, etc.)
        
        Args:
            parquet_path: Path to the complaint_embeddings.parquet file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(parquet_path):
            print(f"Pre-built embeddings file not found: {parquet_path}")
            return False
            
        try:
            print(f"Loading pre-built embeddings from {parquet_path}")
            
            # Check if collection already has data
            count = self.collection.count()
            if count > 0:
                print(f"Collection already contains {count} documents.")
                print("Skipping load - using existing collection.")
                return True
            
            # Read parquet file in batches to handle large files
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(parquet_path)
            total_rows = parquet_file.metadata.num_rows
            batch_size = 10000  # Read 10k rows at a time from parquet
            
            print(f"Total chunks in file: {total_rows:,}")
            print(f"Loading in batches of {batch_size:,}...")
            
            batch_num = 0
            total_loaded = 0
            
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                batch_df = batch.to_pandas()
                batch_num += 1
                
                # Extract data - the parquet uses 'document' for text
                documents = batch_df['document'].tolist()
                
                # Extract embeddings (already as lists)
                embeddings = batch_df['embedding'].tolist()
                
                # Extract metadata from nested struct
                # The metadata column is a dict-like object
                metadatas = []
                ids = []
                
                for idx, row in batch_df.iterrows():
                    # Use the 'id' column if available, otherwise generate
                    doc_id = row.get('id', f"chunk_{total_loaded + idx}")
                    ids.append(str(doc_id))
                    
                    # Extract metadata from the struct
                    meta_dict = row['metadata'] if isinstance(row['metadata'], dict) else {}
                    metadata_flat = {
                        'complaint_id': str(meta_dict.get('complaint_id', '')),
                        'product_category': str(meta_dict.get('product_category', '')),
                        'product': str(meta_dict.get('product', '')),
                        'issue': str(meta_dict.get('issue', '')),
                        'sub_issue': str(meta_dict.get('sub_issue', '')),
                        'company': str(meta_dict.get('company', '')),
                        'state': str(meta_dict.get('state', '')),
                        'date_received': str(meta_dict.get('date_received', '')),
                        'chunk_index': str(meta_dict.get('chunk_index', '')),
                        'total_chunks': str(meta_dict.get('total_chunks', ''))
                    }
                    # Remove empty values
                    metadata_flat = {k: v for k, v in metadata_flat.items() if v}
                    metadatas.append(metadata_flat)
                
                # Add to ChromaDB in smaller batches (ChromaDB has batch size limits)
                chroma_batch_size = 5000
                for i in range(0, len(documents), chroma_batch_size):
                    end_idx = min(i + chroma_batch_size, len(documents))
                    batch_docs = documents[i:end_idx]
                    batch_embeddings = embeddings[i:end_idx]
                    batch_metadatas = metadatas[i:end_idx]
                    batch_ids = ids[i:end_idx]
                    
                    self.collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                
                total_loaded += len(documents)
                print(f"Processed batch {batch_num}: {total_loaded:,}/{total_rows:,} chunks ({total_loaded*100/total_rows:.1f}%)")
            
            final_count = self.collection.count()
            print(f"Successfully loaded {final_count:,} chunks into vector store")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error loading pre-built embeddings: {e}")
            traceback.print_exc()
            return False
    
    def build_from_dataframe(self, df: pd.DataFrame, text_column: str = "clean_narrative"):
        """
        Build vector store from a filtered complaints dataframe.
        
        Args:
            df: DataFrame containing complaint data
            text_column: Column name containing the text to embed
        """
        print("Building vector store from dataframe...")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Process complaints and create chunks
        all_chunks = []
        all_metadata = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            if pd.isna(text) or text.strip() == "":
                continue
                
            chunks = text_splitter.split_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "complaint_id": str(row.get("Complaint ID", "")),
                    "product_category": str(row.get("product_category", "")),
                    "product": str(row.get("Product", "")),
                    "issue": str(row.get("Issue", "")),
                    "sub_issue": str(row.get("Sub-issue", "")),
                    "company": str(row.get("Company", "")),
                    "state": str(row.get("State", "")),
                    "date_received": str(row.get("Date received", "")),
                    "chunk_index": str(chunk_idx),
                    "total_chunks": str(len(chunks))
                })
        
        print(f"Generated {len(all_chunks)} chunks from {len(df)} complaints")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            batch_size=64
        )
        embeddings = embeddings.tolist()
        
        # Add to ChromaDB in batches
        batch_size = 5000
        total = len(all_chunks)
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_docs = all_chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = all_metadata[i:end_idx]
            batch_ids = [f"chunk_{j}" for j in range(i, end_idx)]
            
            self.collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Added batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({end_idx}/{total} chunks)")
        
        print(f"Successfully built vector store with {self.collection.count()} chunks")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the vector store for similar complaint chunks.
        
        Args:
            query_text: User's query/question
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"product_category": "Credit card"})
            
        Returns:
            List of dictionaries containing document, metadata, and distance
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Prepare where clause for filtering
        where_clause = filter_dict if filter_dict else None
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
