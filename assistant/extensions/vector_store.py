import os

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient


class VectorStore:
    _client = None
    _embedder = None
    _vector_store = None

    @classmethod
    def init_app(cls):
        if cls._vector_store is None:
            cls._client = MongoClient(os.environ['MONGODB_URI'])
            cls._embedder = OpenAIEmbeddings(model="text-embedding-3-small")
            db_name = "bp_ai"
            collection_name = "financial_education"
            atlas_vector_search_index_name = "langchain-test-index-vectorstores"

            mongodb_collection = cls._client[db_name][collection_name]

            cls._vector_store = MongoDBAtlasVectorSearch(
                collection=mongodb_collection,
                embedding=cls._embedder,
                index_name=atlas_vector_search_index_name,
                relevance_score_fn="cosine",
            )

    @classmethod
    def get_vector_store(cls):
        if cls._vector_store is None:
            raise RuntimeError("Database not initialized. Call init_app first.")
        return cls._vector_store
