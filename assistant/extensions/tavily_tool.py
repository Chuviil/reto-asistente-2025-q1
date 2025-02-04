import os

from tavily import TavilyClient


class Tavily:
    _client = None

    @classmethod
    def init_app(cls):
        if cls._client is None:
            cls._client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    @classmethod
    def get_tavily_client(cls):
        if cls._client is None:
            raise RuntimeError("Database not initialized. Call init_app first.")
        return cls._client
