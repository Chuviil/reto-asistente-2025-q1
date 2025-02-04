from langchain_openai import ChatOpenAI


class LLM:
    _llm_instance = None

    @classmethod
    def init_app(cls):
        if cls._llm_instance is None:
            cls._llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @classmethod
    def get_llm(cls):
        if cls._llm_instance is None:
            raise RuntimeError("Database not initialized. Call init_app first")
        return cls._llm_instance
