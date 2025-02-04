from flask.views import MethodView
from flask_smorest import Blueprint

from schemas import AssistantSchema
from services.shopping_advisor_service import build_shopping_advisor_graph

blp = Blueprint("shopping-advisor", __name__, description="Shopping advisor")

graph = build_shopping_advisor_graph()


@blp.route("/shopping-advisor")
class ChatRag(MethodView):

    @blp.arguments(AssistantSchema)
    @blp.response(200, AssistantSchema)
    def post(self, request_data):
        response = graph.invoke({"question": request_data["question"]})
        return {"response": response["answer"], "question": request_data["question"]}
