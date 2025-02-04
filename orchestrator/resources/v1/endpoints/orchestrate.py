from typing import TypedDict

from flask.views import MethodView
from flask_smorest import Blueprint

from schemas import OrchestratorSchema
from services.orchestrator_service import build_orchestrator_graph


class State(TypedDict):
    intention: str
    question: str
    guardrail_status: str
    answer: str


blp = Blueprint("Orchestrator", __name__, description="Orchestrator V1")

graph = build_orchestrator_graph()


@blp.route("/orchestrate")
class ChatRag(MethodView):

    @blp.arguments(OrchestratorSchema, location="form")
    @blp.response(200, OrchestratorSchema)
    def post(self, request_data):
        response = graph.invoke({"question": request_data["question"]})
        return {"response": response["answer"], "question": request_data["question"]}
