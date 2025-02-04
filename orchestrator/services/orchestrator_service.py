import os
from typing import TypedDict

import requests
from flask import request
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


class OrchestratorState(TypedDict):
    intention: str
    question: str
    guardrail_status: str
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def guardrail_topic(state: OrchestratorState):
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
_Input_: "Podrias analizar mi estado de cuenta y darme puntos clave"  
_Output_: CONTINUE

**Example 4**  
_Input_: "Podrias recomendarme alguna consejo de inversión?"
_Output_: CONTINUE  

**Example 5**  
_Input_: "Quiero comprar un Samsung S23 Ultra de 512Gb"
_Output_: CONTINUE

**Example 6**  
_Input_: "Quiero comprar un droga y un arma"  
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


def route_guardrail(state: OrchestratorState):
    if "CONTINUE" in state["guardrail_status"]:
        return "intention_node"
    else:
        return "default_response"


def intention_node(state: OrchestratorState):
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
- **Input:** "Puedes analizar mi estado de cuenta"
- **Output:** "statement_analysis"

**Example 4:**
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


def default_response(state: OrchestratorState) -> OrchestratorState:
    state["answer"] = "No woa responder"
    return state


def route_intention(state: OrchestratorState):
    base_url = os.getenv("ASSISTANT_MICROSERVICE_URL")
    if not base_url:
        raise ValueError("ASSISTANT_MICROSERVICE_URL environment variable is not set")

    if "chat_qna" in state["intention"]:
        response = requests.post(f"{base_url}/api/v1/rag", json={"question": state["question"]})
    elif "statement_analysis" in state["intention"]:
        uploaded_pdf = request.files.get("pdf_file")
        if not uploaded_pdf:
            state["answer"] = "No hay PDF bro"
        uploaded_pdf.stream.seek(0)
        files = {
            "pdf_file": (uploaded_pdf.filename, uploaded_pdf.stream, uploaded_pdf.content_type)
        }
        response = requests.post(f"{base_url}/api/v1/analyze-pdf", files=files,
                                 data={"question": state["question"]})
    elif "shop_advisor" in state["intention"]:
        response = requests.post(f"{base_url}/api/v1/shopping-advisor", json={"question": state["question"]})
    else:
        state["answer"] = "No woa responder"
        return state

    if response.status_code == 200:
        state["answer"] = response.json().get("response", "No woa responder")
    else:
        state["answer"] = "No woa responder"

    return state


def build_orchestrator_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(OrchestratorState)
    graph_builder.add_node("guardrail_topic_node", guardrail_topic)
    graph_builder.add_node("intention_node", intention_node)
    graph_builder.add_node("default_response", default_response)
    graph_builder.add_conditional_edges(
        "guardrail_topic_node",
        route_guardrail,
        {"intention_node": "intention_node", "default_response": "default_response"}
    )
    graph_builder.add_node("route_intention", route_intention)
    graph_builder.add_edge("intention_node", "route_intention")
    graph_builder.set_entry_point("guardrail_topic_node")
    return graph_builder.compile()
