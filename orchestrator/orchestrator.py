from flask import Flask
from flask_smorest import Api
from langchain_openai import ChatOpenAI
from resources.orchestrate import blp as OrchestratorBlueprint

app = Flask(__name__)

app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["API_TITLE"] = "Assistant Orchestrator"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.1.0"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

app.config["llm"] = llm

api = Api(app)

api.register_blueprint(OrchestratorBlueprint)
