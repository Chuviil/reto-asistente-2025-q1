from flask import Flask
from flask_smorest import Api

from config.config import Config
from extensions.llm import LLM
from extensions.tavily_tool import Tavily
from extensions.vector_store import VectorStore
from resources.v1.endpoints.analyze_pdf import blp as analyze_pdf_blueprint
from resources.v1.endpoints.chat_rag import blp as chat_rag_blueprint
from resources.v1.endpoints.shopping_advisor import blp as shopping_advisor_blueprint

app = Flask(__name__)

app.config.from_object(Config)

LLM.init_app()
Tavily.init_app()
VectorStore.init_app()

api = Api(app)

api.register_blueprint(chat_rag_blueprint, url_prefix="/api/v1")
api.register_blueprint(shopping_advisor_blueprint, url_prefix="/api/v1")
api.register_blueprint(analyze_pdf_blueprint, url_prefix="/api/v1")
