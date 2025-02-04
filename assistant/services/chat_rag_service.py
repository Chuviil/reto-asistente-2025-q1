from typing import TypedDict, List

from langchain import hub
from langchain_core.documents import Document
from langgraph.constants import START
from langgraph.graph import StateGraph

from extensions.llm import LLM
from extensions.vector_store import VectorStore


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    vector_store = VectorStore.get_vector_store()
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}


def generate(state: State):
    llm = LLM.get_llm()
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_chat_rag_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()
