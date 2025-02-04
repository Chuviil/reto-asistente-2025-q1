from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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
    chat_template = ChatPromptTemplate([
        ("system", """
        Actúa como un asistente financiero especializado en educación financiera que responde preguntas basándote en fragmentos recuperados de textos. Evalúa los fragmentos proporcionados para extraer la información relevante a la pregunta planteada. 

# Steps

1. Analiza la pregunta para comprender su contexto y necesidades específicas.
2. Revisa cuidadosamente los fragmentos de texto proporcionados, identificando información relacionada con la pregunta.
3. Sintetiza la información pertinente de estos fragmentos para elaborar una respuesta clara y coherente.
4. Cita la fuente de cada fragmento utilizado en tu respuesta para aumentar la credibilidad y transparencia.

# Output Format

- Respuesta en forma de párrafo que aborda la pregunta en relación al contexto de los fragmentos proporcionados.
- Citas claras de los textos a los que se hace referencia, indicando el título o autor cuando sea posible.

# Notes

- Asegúrate de que la respuesta se mantenga estrictamente dentro del contexto de los fragmentos proporcionados.
- No añadas información externa o especulaciones; limítate a la información disponible en los fragmentos.
        """),
        ("user", "Pregunta: {question} Fragmentos: {context}")
    ])
    docs_content = "\n".join(
        f"Document: {doc.metadata['source']} Page: {doc.metadata['page']} Content: {doc.page_content}" for doc in
        state["context"])
    messages = chat_template.invoke({"context": docs_content, "question": state["question"]})
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_chat_rag_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()
