from src.yt_rag.llm.graph.graph_state import AgentState
from src.yt_rag.llm.tool.tool_function import rag_search


def rag_agent(state: AgentState):
    print("RAG Agent called.")
    step_index = state["current_step_index"]
    agent_plan_metadata = state["detail_plan"][step_index]
    agent_goal = agent_plan_metadata["description"]
    dependent_results = agent_plan_metadata["depends_on_plan_ids"]
    prev_result_context = []
    for result_id in dependent_results:
        prev_result_context.append(state[result_id])
    rag_parameter = {"user_query": agent_goal}
    model_response = rag_search(arg=rag_parameter, collection_id=state["collection_id"])
    state["results"][step_index + 1] = str(model_response)
    state["final_output"] = model_response
    state["current_step_index"] += 1
    return state
