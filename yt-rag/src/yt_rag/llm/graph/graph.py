from langgraph.graph import StateGraph, END
from src.yt_rag.llm.graph.graph_state import AgentState
from src.yt_rag.llm.agents.chat import chat_agent
from src.yt_rag.llm.agents.fallback import fallback_agent
from src.yt_rag.llm.agents.rag import rag_agent
from src.yt_rag.llm.agents.planner import planner_node
from src.yt_rag.llm.agents.supervisor import supervisor_node
from src.yt_rag.llm.agents.planner import plan_executioner

graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("planner", planner_node)
graph.add_node("chat", chat_agent)
graph.add_node("fallback", fallback_agent)
graph.add_node("rag", rag_agent)
graph.add_node("executioner", plan_executioner)

graph.set_entry_point("supervisor")


def router_node(state: AgentState):
    return state["current_agent"]


graph.add_conditional_edges(
    "supervisor",
    router_node,
    {
        "planner": "planner",
        "chat": "chat",
        "fallback": "fallback",
    },
)


def conditional_router(state: AgentState):
    if state["current_step_index"] > len(state["detail_plan"]):
        print("stopping")
        return "stop"

    plan_step_index = state["current_step_index"]
    state["current_agent"] = state["detail_plan"][plan_step_index]["agent_name"]
    return state["current_agent"]


graph.add_conditional_edges(
    "executioner",
    conditional_router,
    {
        "rag": "rag",
        "chat": "chat",
        "fallback": "fallback",
        "stop": END,
    },
)
graph.add_edge("planner", "executioner")
graph.add_edge("chat", END)
graph.add_edge("fallback", END)
graph.add_edge("rag", "executioner")

app = graph.compile()
