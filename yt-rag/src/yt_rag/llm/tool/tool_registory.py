from src.yt_rag.llm.tool.tool_function import rag_search


class ToolRegistory:

    TOOL_MAPPING = {"retrieve_context_chunks": rag_search}
