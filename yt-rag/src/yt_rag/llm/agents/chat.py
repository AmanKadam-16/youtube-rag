from src.yt_rag.llm.graph.graph_state import AgentState
from src.yt_rag.core.llm.config import call_llm


def chat_agent(state: AgentState):
    print("Chat Agent Invoked.")
    prev_result_context = []
    step_index = state.get("current_step_index", 0)
    user_query = state["user_input"]
    agent_goal = state["user_input"]
    if len(state["detail_plan"]) > 0:
        agent_plan_metadata = state["detail_plan"][step_index]
        agent_goal = agent_plan_metadata["description"]
        dependent_results = agent_plan_metadata["depends_on_plan_ids"]
        for result_id in dependent_results:
            prev_result_context.append(state["results"][result_id])
    system = f"""
                    You are ChatAgent. Be friendly and conversational.
                    User Query/Message:
                    {user_query}

                    Agent Executed Context:
                    {prev_result_context}

                    Note: If the Executed Context is N/A or emtpy and user input was not a greeting then provide message of denial to the user.
                    CITATION: If you use any information from the context, cite the sources as [source1], [source2], etc. Always provide citations (Actual source info) at the end of your response.
                    INSTRUCTION: While responding directly start with the answer. Don't start with - `Sure thing! Here's a friendly, easy-to-read rundown of what those snippets are talking about:`
                """
    model_response = call_llm(system, agent_goal)
    state["results"][step_index + 1] = model_response
    state["final_output"] = model_response
    return state
