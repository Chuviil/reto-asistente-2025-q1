import io

import pdfplumber
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint

from schemas import ChatRagSchema


def table_to_markdown(table):
    if not table:
        return ""

    header, *rows = table

    header = [cell if cell is not None else "" for cell in header]
    rows = [[cell if cell is not None else "" for cell in row] for row in rows]

    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def pdf_to_text_with_tables(pdf_path):
    output_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            output_lines.append(f"## Page {page_number}\n")

            page_text = page.extract_text()
            if page_text:
                output_lines.append(page_text)
                output_lines.append("\n")

            tables = page.extract_tables()
            if tables:
                output_lines.append("### Detected Table(s):\n")
                for i, table in enumerate(tables, start=1):
                    output_lines.append(f"**Table {i}:**")
                    md_table = table_to_markdown(table)
                    output_lines.append(md_table)
                    output_lines.append("\n")

    return "\n".join(output_lines)


blp = Blueprint("analyze_pdf", __name__, description="Bank Statement analysis")


@blp.route("/analyze-pdf")
class ChatRag(MethodView):

    @blp.arguments(ChatRagSchema, location="form")
    @blp.response(200, ChatRagSchema)
    def post(self, request_data):
        pdf_file = request.files.get("pdf_file")
        if pdf_file is None:
            return {"message": "Missing pdf_file in request"}, 400

        pdf_stream = io.BytesIO(pdf_file.read())

        pdf_content = pdf_to_text_with_tables(pdf_stream)

        return {"response": pdf_content, "question": request_data["question"]}
