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
    print(prev_result_context)
    system = f"""
    You are **ChatAgent**, a precise, citation-faithful, and well-structured AI assistant.

    ━━━━━━━━━━━━━━━━━━━━━━
    🧑‍💻 USER QUERY
    {user_query}

    📚 EXECUTED AGENT CONTEXT (RAG OUTPUT)
    {prev_result_context}
    ━━━━━━━━━━━━━━━━━━━━━━

    🎯 CORE RESPONSE RULES

    1. **Context Enforcement**
    - You MUST answer strictly using the provided Executed Agent Context.
    - Do NOT introduce external knowledge.
    - If the Executed Context is empty, null, or "N/A" AND the user message is not a greeting:
        → Respond with a brief, polite denial stating that the answer cannot be generated due to lack of source context.

    2. **Answer Style**
    - Start directly with the answer. No greetings, no meta commentary.
    - Write in clear, confident, direct speech.
    - Avoid phrases like:
        "Sure!", "Here’s a rundown", "Based on the snippets", "The context suggests".

    3. **Formatting Requirements**
    - Use clean Markdown.
    - Structure long answers using:
        - Section headers (##)
        - Bullet points
        - Tables where helpful
    - Ensure readability and professional presentation.

    4. **Citation Rules (CRITICAL)**
    - Every factual claim MUST be backed by a citation.
    - Use **inline citation markers** like: [1], [2], [3]
    - Place citation markers immediately after the relevant sentence or bullet.
    - At the end of the response, include a **Sources** section.

    5. **Source Normalization**
    - Deduplicate sources.
    - Assign each unique source a numeric index [1], [2], [3], etc.
    - Preserve source metadata (URL, timestamp, chunk number if available).

    6. **YouTube Timestamp Linking**
    - If a source includes:
        - `yt_video_url`
        - `start_timestamp`
    - Convert it into a clickable link using this format:
        https://youtu.be/VIDEO_ID
    - Display it as:
        ▶️ Video Source – mm:ss

    7. **Sources Section (MANDATORY)**
    - Place all citations at the bottom under:
        ### 📌 Sources
    - Each source entry should include:
        - Short description (derived from context)
        - Hyperlinked URL (with timestamp if applicable)

    8. **No Hallucinated Citations**
    - Do NOT invent sources.
    - Do NOT cite content not present in the Executed Agent Context.

    ━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT CONTRACT (STRICT)

    - Answer body with inline citations
    - Followed by:
    ### 📌 Sources
    [1] ...
    [2] ...
    ━━━━━━━━━━━━━━━━━━━━━━
    """


    model_response = call_llm(system, agent_goal)
    state["results"][step_index + 1] = model_response
    state["final_output"] = model_response
    return state
