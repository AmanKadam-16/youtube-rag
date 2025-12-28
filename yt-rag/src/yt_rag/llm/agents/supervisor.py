from src.yt_rag.llm.graph.graph_state import AgentState
from src.yt_rag.core.llm.config import call_agentic_llm


def supervisor_node(state: AgentState):
    print("=> Supervisor Agent Invoked")
    sys_prompt = """
    You are a supervisor agent that decides which agent should handle a user query.

    Return ONLY ONE word:
    chat | fallback | planner

    Rules:
    - Do NOT explain, justify, or answer the user.
    - Return a single word only, without quotes or punctuation.
    - Pick fallback if the query cannot be handled by any of the available agents.

    Agent Info:
    chat : Handles casual messages, greetings, or trivial queries. Examples: "Hi", "Hello", "How are you?".
    fallback : Pick when queries are out-of-scope or cannot be handled by chat or planner.
    planner : Pick this when the user query requires actionable steps that can be executed via other agents (rag_agent, chat_agent).

    NOTE:
    If user query is not a greeting but a genuine question then go for planner.
    """
    user_obj = {"role": "user", "content": state["user_input"]}
    sys_obj = {"role": "system", "content": sys_prompt}
    final_messages = [sys_obj, user_obj]

    llm_response = call_agentic_llm(messages=final_messages)
    state["current_agent"] = llm_response
    print("This was Supervisor Agent Choice ->> ", llm_response)
    return state
