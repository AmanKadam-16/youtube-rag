from fastapi.routing import APIRouter
from src.yt_rag.schema import chatbot as chatbot_schema
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from src.yt_rag.utils.util import export_mermaid_graph
from src.yt_rag.llm.graph.graph import app as planner_agentic_graph_app



router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post("/planner-agent-conversate", response_model=chatbot_schema.ChatResponse)
def plan_and_execute(user_input: chatbot_schema.ChatInput):
    try:
        payload = {
            "user_input": user_input.user_prompt,
            "results": {},
            "current_step_index": 0,
            "collection_id": user_input.collection_id,
            "detail_plan": [],
        }
        export_mermaid_graph(
            app=planner_agentic_graph_app, output_path="graph_01_version.mmd"
        )
        service_response = planner_agentic_graph_app.invoke(payload)
        formatted_result = {"response": service_response["final_output"]}
        result = JSONResponse(status_code=200, content=formatted_result)
        return result
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code, content=f"HTTP Exception {he.detail}"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=f"Exception {e}")
    