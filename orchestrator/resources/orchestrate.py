from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from schemas import OrchestratorSchema


from typing import TypedDict, NotRequired

class State(TypedDict):
    intention: str
    question: str
    guardrail_status: str
    answer: str


blp = Blueprint("orchestrate", __name__, description="Orchestrator")


def guardrail_topic(state: State):
    llm: ChatOpenAI = current_app.config["llm"]
    chat_template = ChatPromptTemplate(
        [("system",
          """
          Evaluate the user input to determine if it adheres to the allowed topics and conditions. The user input should only pertain to financial education questions, bank statement analysis, or product recommendations. Ensure compliance with privacy and legality constraints.

- **Allowed Topics**:
  - Financial education questions
  - Bank statement analysis
  - Product recommendations

- **Constraints**:
  - Bank Statement Analysis: User input must not compromise sensitive client data or request information such as a client's earnings.
  - Product Recommendations: User input must not involve illegal or criminal activities, including requests for products such as guns or drugs.

# Steps

1. **Identify Topic**: Determine if the user input involves financial education, bank statement analysis, or product recommendations.
2. **Verify Compliance**: Check the input against the specified constraints based on its identified topic:
   - For bank statement analysis, ensure no sensitive data requests.
   - For product recommendations, ensure no criminal activities are involved.
3. **Decide Response**: Based on the checks, decide if the interaction can continue or should be stopped.

# Output Format

Provide a response using either "CONTINUE" or "STOP."

# Examples

**Example 1**  
_Input_: "Me puedes dar tips para ahorrar de forma efectiva?"  
_Output_: CONTINUE  

**Example 2**  
_Input_: "Basa en la informacion del estado de cuenta, cual es el valor aproximado de ingresos mensuales de este cliente?"  
_Output_: STOP  

**Example 3**  
_Input_: "Podrias recomendarme alguna consejo de inversión?"
_Output_: CONTINUE  

**Example 4**  
_Input_: "Quiero comprar droga"  
_Output_: STOP

**Example 5**  
_Input_: "Quiero comprar un arma"  
_Output_: STOP

# Notes

- Always err on the side of caution; if in doubt about the content, opt for "STOP."
- Consider any ambiguous requests or requests with implied sensitive or illegal content as non-compliant.
        """),
         ("user", "{question}")]
    )
    messages = chat_template.invoke({"question": state["question"]})
    response = llm.invoke(messages)
    state["guardrail_status"] = response.content
    return state


def route_guardrail(state: State):
    if "CONTINUE" in state["guardrail_status"]:
        return "intention_node"
    else:
        return "default_response"


def intention_node(state: State):
    llm: ChatOpenAI = current_app.config["llm"]
    chat_template = ChatPromptTemplate([
        ("system",
         """
         Classify the user's input into one of three defined intentions based on the content provided. The possible intentions are: "Financial Education Q&A," "Bank Statement Analysis," and "Shopping Advisor."

- "Financial Education Q&A": If the input includes a financial education question.
- "Bank Statement Analysis": If the input includes a request to analyze a bank statement.
- "Shopping Advisor": If the input refers to buying or wanting something.

# Steps

1. Analyze the user's input text.
2. Determine which of the three categories the input best fits.
3. Consider keywords and context to accurately classify the intention.
4. Respond with the appropriate classification label.

# Output Format

Respond with one of the following terms, based on the detected intention:
- "chat_qna" for Financial Education Q&A.
- "statement_analysis" for Bank Statement Analysis.
- "shop_advisor" for Shopping Advisor.

# Examples

**Example 1:**
- **Input:** "Tips para ahorrar de forma eficiente"
- **Output:** "chat_qna"

**Example 2:**
- **Input:** "Puedes ayudarme a revisar mis gastos bancarios del mes pasado?"
- **Output:** "statement_analysis"

**Example 3:**
- **Input:** "Quiero comprarme un Samsung S24 Ultra"
- **Output:** "shop_advisor"

# Notes

- Ensure to only use the specified output terms.
- The classifications should be mutually exclusive and comprehensive. Always choose the most fitting label for each input.
         """),
        ("user", "{question}")
    ])
    messages = chat_template.invoke({"question": state["question"]})
    response = llm.invoke(messages)
    state["intention"] = response.content
    return state

def default_response(state: State):
    state["answer"] = "No woa responder"
    return state

#TODO: Call microservices
def route_intention(state: State):
    if "chat_qna" in state["intention"]:
        state["answer"] = "chat_rag"
    elif "statement_analysis" in state["intention"]:
        state["answer"] = "pdf_analysis"
    elif "shop_advisor" in state["intention"]:
        state["answer"] = "shop_advisor"
    else:
        state["answer"] = "No woa responder"
    return state


graph_builder = StateGraph(State)
graph_builder.add_node("guardrail_topic_node", guardrail_topic)
graph_builder.add_node("intention_node", intention_node)
graph_builder.add_node("default_response", default_response)
graph_builder.add_conditional_edges("guardrail_topic_node", route_guardrail, {"intention_node": "intention_node", "default_response": "default_response"})
graph_builder.add_node("route_intention", route_intention)
graph_builder.add_edge("intention_node", "route_intention")
graph_builder.set_entry_point("guardrail_topic_node")
graph = graph_builder.compile()


@blp.route("/orchestrate")
class ChatRag(MethodView):

    @blp.arguments(OrchestratorSchema)
    @blp.response(200, OrchestratorSchema)
    def post(self, request_data):
        response = graph.invoke({"question": request_data["question"]})
        return {"response": response["answer"], "question": request_data["question"]}
