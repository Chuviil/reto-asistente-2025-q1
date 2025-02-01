from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint
from langchain import hub
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from schemas import ChatRagSchema

blp = Blueprint("chat_rag", __name__, description="Chat with RAG")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    vector_store: MongoDBAtlasVectorSearch = current_app.config['vector_store']
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}


def generate(state: State):
    llm: ChatOpenAI = current_app.config['llm']
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


@blp.route("/rag")
class ChatRag(MethodView):

    @blp.arguments(ChatRagSchema)
    @blp.response(200, ChatRagSchema)
    def post(self, request_data):
        response = graph.invoke({"question": request_data["question"]})
        return {"response": response["answer"], "question": request_data["question"]}
