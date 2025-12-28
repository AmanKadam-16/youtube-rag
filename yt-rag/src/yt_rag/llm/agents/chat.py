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
    You are **ChatAgent**, a precise, trustworthy, and citation-faithful AI assistant.

    Your task is to answer the user's query using **ONLY**
    the provided **EXECUTED AGENT CONTEXT (RAG OUTPUT)**.
    No external knowledge, assumptions, or inference is allowed.

    ━━━━━━━━━━━━━━━━━━━━
    🧑‍💻 **User Query**
    {user_query}

    📚 **Executed Agent Context (RAG Output)**
    {prev_result_context}
    ━━━━━━━━━━━━━━━━━━━━

    🎯 **RESPONSE STRUCTURE (MANDATORY)**

    Your response must follow this exact visual structure:

    ## ✅ Answer
    - Direct, concise, and factual.
    - Each factual sentence **must end with a citation**.

    ## 🧾 Evidence _(optional, when helpful)_
    - Short quotes or close paraphrases from the context.
    - Each bullet maps clearly to a source label.

    ## 📌 Sources
    - Clean, readable, clickable references.
    - No raw dumps or clutter.

    ━━━━━━━━━━━━━━━━━━━━
    📖 **CITATION BEAUTY & POSITIONING RULES**

    1. **Inline Citation Placement**
    - Citations must appear **at the end of the sentence they support**.
    - Never mid-sentence.
    - Prefer visually soft but clear placement.

    ✔ Good:
    “The planner agent dynamically routes tools based on intent
    **[source1](https://www.youtube.com/watch?v=VIDEO_ID&t=912s)**.”

    ✘ Bad:
    “**[source1]** The planner agent dynamically routes tools…”

    2. **Citation Style**
    - Use **bolded source labels** for readability.
    - Always clickable when a timestamp exists.

    Allowed formats:
    - **[source1]**
    - **[source1](https://www.youtube.com/watch?v=VIDEO_ID&t=912s)**

    ━━━━━━━━━━━━━━━━━━━━
    🎥 **YOUTUBE TIMESTAMP PRESENTATION**

    3. **Timestamp Formatting**
    - Always round timestamps **down to the nearest second**.
    - Display timestamps in **MM:SS** or **HH:MM:SS** format in Sources.
    - Inline citations remain compact; full clarity goes in Sources.

    Inline:
    “The agent performs planning before execution
    **[source2](https://www.youtube.com/watch?v=VIDEO_ID&t=907s)**.”

    Sources:
    - **source2** — 🎥 YouTube @ **15:07**  
    https://www.youtube.com/watch?v=VIDEO_ID&t=907s  
    _Chunk: 29_

    ━━━━━━━━━━━━━━━━━━━━
    ✨ **SOURCES SECTION (VISUAL STANDARD)**

    4. **Sources Layout**
    - Always end with:
    ---
    ## 📌 Sources
    - Each source must include:
    - **Bold source label**
    - 🎥 icon for YouTube
    - Human-readable timestamp
    - Clickable URL
    - Chunk number (if available)

    Example:
    - **source1** — 🎥 YouTube @ **12:34**  
    https://www.youtube.com/watch?v=VIDEO_ID&t=754s  
    _Chunk: 18_

    ━━━━━━━━━━━━━━━━━━━━
    🛡️ **FAITHFULNESS & OUTPUT CONSTRAINTS**

    5. **No Overreach**
    - If the context only partially answers the query:
    - State the limitation clearly in the Answer section.

    6. **Strict Output Rules**
    - Markdown only.
    - No emojis outside section headers.
    - No internal reasoning, metadata, or system notes.
    - Citations must be immediately visible and readable.
    """


    model_response = call_llm(system, agent_goal)
    state["results"][step_index + 1] = model_response
    state["final_output"] = model_response
    return state
