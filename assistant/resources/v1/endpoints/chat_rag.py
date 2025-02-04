from flask.views import MethodView
from flask_smorest import Blueprint

from schemas import AssistantSchema
from services.chat_rag_service import build_chat_rag_graph

blp = Blueprint("chat_rag", __name__, description="Chat with RAG")

graph = build_chat_rag_graph()


@blp.route("/rag")
class ChatRag(MethodView):

    @blp.arguments(AssistantSchema)
    @blp.response(200, AssistantSchema)
    def post(self, request_data):
        response = graph.invoke({"question": request_data["question"]})
        return {"response": response["answer"], "question": request_data["question"]}
