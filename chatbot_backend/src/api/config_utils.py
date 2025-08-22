import os
from functools import lru_cache
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Pinecone
    PINECONE_API_KEY: str = Field(default="", description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(default="", description="Pinecone index name (if not using host)")
    PINECONE_NAMESPACE: str = Field(default="default", description="Pinecone namespace for multi-tenancy")
    # For serverless connection provide host; for legacy/provisioned, provide environment and index_name
    PINECONE_HOST: str = Field(default="", description="Pinecone index host URL (serverless)")
    PINECONE_ENVIRONMENT: str = Field(default="", description="Pinecone environment (legacy/provisioned)")

    # Retrieval settings
    PINECONE_TOP_K: int = Field(default=3, description="Top K results for semantic search")

    # Embeddings
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key (optional in local dev)")
    EMBEDDING_DIM: int = Field(default=768, description="Embedding dimension (text-embedding-004 uses 768)")

    # Chunking
    CHUNK_SIZE: int = Field(default=1200, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks in characters")

    # CORS / App
    APP_ENV: str = Field(default="development")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Read from environment variables or .env loaded by the runtime
    return Settings(
        PINECONE_API_KEY=os.getenv("PINECONE_API_KEY", ""),
        PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME", ""),
        PINECONE_NAMESPACE=os.getenv("PINECONE_NAMESPACE", "default"),
        PINECONE_HOST=os.getenv("PINECONE_HOST", ""),
        PINECONE_ENVIRONMENT=os.getenv("PINECONE_ENVIRONMENT", ""),
        PINECONE_TOP_K=int(os.getenv("PINECONE_TOP_K", "3")),

        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
        EMBEDDING_DIM=int(os.getenv("EMBEDDING_DIM", "768")),

        CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "1200")),
        CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "200")),

        APP_ENV=os.getenv("APP_ENV", "development"),
    )
