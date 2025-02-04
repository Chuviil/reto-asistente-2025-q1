from flask import Flask
from flask_smorest import Api

from config.config import Config
from extensions.llm import LLM
from resources.v1.endpoints.orchestrate import blp as orchestrator_blueprint

app = Flask(__name__)

app.config.from_object(Config)

api = Api(app)

LLM.init_app()

api.register_blueprint(orchestrator_blueprint, url_prefix="/api/v1")
