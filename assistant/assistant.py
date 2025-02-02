import os

from flask import Flask
from flask_smorest import Api
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from tavily import TavilyClient

from resources.chat_rag import blp as ChatRagBlueprint
from resources.shopping_advisor import blp as ShoppingAdvisorBlueprint

app = Flask(__name__)

app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["API_TITLE"] = "Assistant Microservice"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.1.0"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

client = MongoClient(os.environ['MONGODB_URI'])

DB_NAME = "bp_ai"
COLLECTION_NAME = "financial_education"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embedder,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

app.config["vector_store"] = vector_store
app.config["llm"] = llm
app.config["tavily_client"] = tavily_client

api = Api(app)

api.register_blueprint(ChatRagBlueprint)
api.register_blueprint(ShoppingAdvisorBlueprint)
