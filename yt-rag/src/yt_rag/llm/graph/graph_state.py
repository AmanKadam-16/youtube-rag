from typing import TypedDict, Dict, Any


class Plan(TypedDict):
    id: int
    description: str
    agent_name: str
    depends_on_plan_ids: list[int]


class AgentState(TypedDict):
    user_input: str
    goal: str
    detail_plan: list[Plan]
    results: Dict[int, Any]
    collection_id: str
    current_agent: str
    current_step_index: int
    final_output: str
