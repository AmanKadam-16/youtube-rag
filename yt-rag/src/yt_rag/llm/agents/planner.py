from src.yt_rag.llm.graph.graph_state import AgentState
from src.yt_rag.core.llm.config import call_agentic_llm
import json


def planner_node(state: AgentState):
    print("=> Planner Agent Invoked")
    sys_prompt = """
    You are  a Planner Agent.
    Your task is to analyse user input and only just plan the flow of 
    execution and selection of Agents available.
    
    The Agents that we have are as follows:
    chat : Used to rephrase the structured result after final execution into friendly, readable and represent in natural language.
    fallback : Can be used when user query can't be handled with current agents or user query is out of scope or irrelevant.
    rag : Can be used to pull context (chunks of context) from knowledge base as per user query. (Choose when user query is not a greeting but a genuine question.)
    
    Your Response should Strictly should be in following JSON format:
    {{
        goal: "rephrased actual goal of user input.",
        plan_flow: [
            {{
                id: int // index of plan (start from 1) 
                description: str // Description or Narrow Goal for that specific Agent to achieve. // In case of RAG agent the description for RAG agent should be rephrased version of User Query for efficient retrival.
                agent_name: str // agent name (strictly out of given agent names.)
                depends_on_plan_ids: list[int] // id of plan on who's input the current agent execution depends. 
            }}
        ]
    }}
    """
    user_obj = {"role": "user", "content": state["user_input"]}
    sys_obj = {"role": "system", "content": sys_prompt}
    final_messages = [sys_obj, user_obj]

    llm_response = call_agentic_llm(messages=final_messages)
    parsed_llm_response = json.loads(llm_response)
    state["goal"] = parsed_llm_response["goal"]
    state["detail_plan"] = parsed_llm_response["plan_flow"]
    print(f"Plan of Action => \n\n{parsed_llm_response}\n\n")
    return state


def plan_executioner(state: AgentState):
    print("Execution Node invoked.")
    plan_metadata = state["detail_plan"]
    if plan_metadata:
        step_id = state["current_step_index"]
        state["current_agent"] = plan_metadata[step_id]["agent_name"]
    return state
