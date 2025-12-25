class ToolsList:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_context_chunks",
                "description": (
                    "Retrieves the most relevant contextual chunks from the knowledge base "
                    "using a rephrased and normalized user query. Intended for RAG pipelines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": (
                                "A rewritten, search-optimized version of the user's original message. "
                                "Should remove ambiguity, resolve pronouns, and include key terms "
                                "needed to retrieve relevant context."
                            ),
                        }
                    },
                    "required": ["user_query"],
                },
            },
        }
    ]
