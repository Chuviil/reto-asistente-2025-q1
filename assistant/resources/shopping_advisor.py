from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from schemas import ChatRagSchema

blp = Blueprint("shopping-advisor", __name__, description="Shopping advisor")


class State(TypedDict):
    question: str
    product: str
    context: List[Document]
    answer: str


def extract_product(state: State):
    llm: ChatOpenAI = current_app.config['llm']
    prompt_template = ChatPromptTemplate([
        ("system",
         """
         Extract the product name and relevant details from the user's input to form a detailed search query.

Consider details such as brand, model, size, color, specifications, or any distinguishing features. Ensure the query is comprehensive and tailored for searches in Ecuador.

# Steps

1. **Identify Key Details**: Extract product name, brand, model, size, color, specifications, and distinguishing features.
2. **Formulate Query**: Combine these details into a coherent search query in Spanish.
3. **Location Tailoring**: Ensure to include "Ecuador" in the search query to tailor the results to this location.

# Output Format

Formulate the search query as a single-line sentence in Spanish. Ensure the query includes the product details and ends with 'Ecuador'.

# Examples

**Input**: "I am looking for a Samsung Galaxy S21 smartphone in Phantom Gray with 256GB storage."

**Output**: Donde comprar un Samsung Galaxy S21 smartphone Phantom Gray 256GB en Ecuador

**Input**: "Necesito una bicicleta de montaña Trek modelo Marlin 5, color rojo y talla M."

**Output**: Donde comprar una bicicleta de montaña Trek Marlin 5 roja talla M  en Ecuador

# Notes

- If the user does not specify certain details such as color or size, include only the available information in the query.
- Focus on the specificity and accuracy of the extracted information.
- Ensure that the translation to Spanish maintains the context and meaning of the query.
         """),
        ("user", "The users wants: {user_input}")
    ])
    messages = prompt_template.invoke({"user_input": state["question"]})
    response = llm.invoke(messages)
    state["product"] = response.content
    return state


def search_product(state: State):
    print(f"Se va a buscar: {state['product']}")
    tavily_client = current_app.config["tavily_client"]
    search_results = tavily_client.search(state["product"])
    print(f"Resultados de busqueda: {search_results}")
    state["context"] = search_results["results"]
    return state


def analyze_results(state: State):
    llm: ChatOpenAI = current_app.config['llm']
    prompt_template = ChatPromptTemplate([
        ("system",
         """
         Analyze the given product desired by the user and a list of search results. Identify and extract the top 5 best results based on user preference and present them in a markdown table format.

# Steps

1. **Understand User Preference**: Determine the key attributes of the product that are important for the user.
2. **Analyze Search Results**: Review the provided list of search results, focusing on attributes that match the user's preferences.
3. **Selection Criteria**: Evaluate each result considering features like price, reliability of the seller, and availability.
4. **Format Output**: Select the top 5 results and organize the information into a markdown table with specified columns.

# Output Format

Provide a Markdown table with the following columns:
- **Nombre Artículo**: Name of the item.
- **Comercio**: The seller's platform or store name.
- **Precio en USD**: Price in USD.
- **Web del anuncio**: URL of the listing.

# Examples

**Input:**
- User wants a 'Wireless Mouse'.
- Results list: A set of search results including product names, sellers, prices, and URLs.

**Output in Markdown:**

| Nombre Artículo     | Comercio        | Precio en USD | Web del anuncio        |
|---------------------|-----------------|---------------|------------------------|
| [Example Mouse 1]   | [Example Store] | [XX USD]      | [https://example.com1] |
| [Example Mouse 2]   | [Example Store] | [YY USD]      | [https://example.com2] |
| [Example Mouse 3]   | [Example Store] | [ZZ USD]      | [https://example.com3] |
| [Example Mouse 4]   | [Example Store] | [AA USD]      | [https://example.com4] |
| [Example Mouse 5]   | [Example Store] | [BB USD]      | [https://example.com5] |

(Ensure that all real examples provide accurate and complete data corresponding to the product and links.)

# Notes

- Ensure the prices are compared accurately and reflect the latest available information.
- Consider the seller's reliability and the product's specifications that match user needs.
- It's vital to verify that the URLs are valid and lead directly to the product's page.
         """),
        ("user", "The users wants: {product}, Results found on Internet: {products}")
    ])
    messages = prompt_template.invoke({"product": state["question"], "products": state["context"]})
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state


graph_builder = StateGraph(State).add_sequence([extract_product, search_product, analyze_results])
graph_builder.add_edge(START, "extract_product")
graph = graph_builder.compile()


@blp.route("/shopping-advisor")
class ChatRag(MethodView):

    @blp.arguments(ChatRagSchema)
    @blp.response(200, ChatRagSchema)
    def post(self, request_data):
        response = graph.invoke({"question": request_data["question"]})
        return {"response": response["answer"], "question": request_data["question"]}
