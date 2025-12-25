def fallback_agent(state):
    print("Fallback node invoked.")
    state["final_output"] = "Sorry, IDK."
    return state
