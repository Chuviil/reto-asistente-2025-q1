from io import BytesIO

from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint

from schemas import AssistantSchema
from services.analyze_pdf_service import build_pdf_analyzer_graph

blp = Blueprint("analyze_pdf", __name__, description="Bank Statement analysis")

graph = build_pdf_analyzer_graph()


@blp.route("/analyze-pdf")
class ChatRag(MethodView):

    @blp.arguments(AssistantSchema, location="form")
    @blp.response(200, AssistantSchema)
    def post(self, request_data):
        pdf_file = request.files.get("pdf_file")
        if pdf_file is None:
            return {"message": "Missing pdf_file in request"}, 400

        pdf_stream = BytesIO(pdf_file.read())

        response = graph.invoke({"pdf_stream": pdf_stream, "question": request_data["question"]})

        return {"response": response["answer"], "question": request_data["question"]}
