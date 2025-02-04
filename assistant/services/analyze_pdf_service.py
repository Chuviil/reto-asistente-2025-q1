from io import BytesIO
from typing import TypedDict

import pdfplumber
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from extensions.llm import LLM


class State(TypedDict):
    pdf_stream: BytesIO
    bank_statement: str
    answer: str


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


def pdf_to_text_with_tables(state: State):
    output_lines = []
    with pdfplumber.open(state["pdf_stream"]) as pdf:
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

    state["bank_statement"] = "\n".join(output_lines)
    return state


def analyze_bank_statement(state: State):
    llm = LLM.get_llm()
    prompt_template = ChatPromptTemplate([
        ("system",
         """
         Analyze a bank statement to identify financial trends and patterns. Specifically, extract and list the top 5 biggest individual expenses ordered in descending order, and identify concurrent expenses by establishment to determine the top 3 biggest concurrent expenses. The response should be provided in Spanish.

# Steps

1. **Extract All Expenses:** Go through the bank statement and extract each expense and its amount.
2. **Identify Top Individual Expenses:** 
   - Sort expenses in descending order based on amount.
   - Select the top 5 expenses.
3. **Identify Concurrent Expenses:**
   - Group expenses by the establishment name.
   - Calculate the total amount for each group.
   - Sort these grouped expenses in descending order based on total amount.
   - Select the top 3 grouped concurrent expenses.
4. **Translate and Format in Spanish:** Ensure the final results are formatted in Spanish, and organized from greatest to least.

# Output Format

- A list of the top 5 biggest individual expenses.
- A grouped list of expenses by establishment.
- The top 3 biggest concurrent (grouped) expenses.
- All outputs should be structured in Spanish without English explanations.

# Examples

**Entrada del extracto bancario:**
- Compra A: 700 USD
- Compra B: 650 USD
- Compra A: 500 USD
- Compra C: 450 USD
- Compra B: 300 USD
- Compra C: 250 USD
- Compra D: 100 USD

**Salida esperada (abreviada, los ejemplos reales deben incluir más detalles en español):**

- **Las 5 mayores gastos individuales:** 
  1. Compra D: 700 USD
  2. Compra C: 550 USD
  3. Compra B: 450 USD
  4. Compra A: 300 USD
  5. Compra B: 200 USD

- **Gastos concurrentes agrupados por establecimiento:**
  - Compra A: 800 USD
  - Compra B: 550 USD
  - Compra C: 760 USD
  - Compra D: 700 USD

- **Las 3 mayores gastos concurrentes por establecimiento:**
  1. Compra B: 850 USD
  2. Compra C: 830 USD
  3. Compra D: 700 USD

# Notes

- Ensure all numerical and monetary values are processed accurately.
- The names and amounts should be checked for correct groupings before the final evaluation.
- The output must be solely in Spanish and formatted concisely.
         """),
        ("user", "Bank Statement: {bank_statement}")
    ])
    messages = prompt_template.invoke({"bank_statement": state["bank_statement"]})
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state


def build_pdf_analyzer_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State).add_sequence([pdf_to_text_with_tables, analyze_bank_statement])
    graph_builder.add_edge(START, "pdf_to_text_with_tables")
    return graph_builder.compile()
