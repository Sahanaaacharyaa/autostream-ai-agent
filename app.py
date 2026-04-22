import streamlit as st
from langgraph.graph import StateGraph, END

from graph import detect_intent, respond, lead_flow, route


st.set_page_config(page_title="AutoStream AI Agent")

st.title("🎬 AutoStream AI Agent")


# ---------------- STATE ---------------- #

if "state" not in st.session_state:
    st.session_state.state = {
        "user_input": "",
        "intent": None,
        "response": None,
        "chat": [],
        "name": None,
        "email": None,
        "platform": None,
        "plan": None,
        "collecting_lead": False
    }


# ---------------- GRAPH ---------------- #

builder = StateGraph(dict)

builder.add_node("intent", detect_intent)
builder.add_node("respond", respond)
builder.add_node("lead", lead_flow)

builder.set_entry_point("intent")

builder.add_conditional_edges(
    "intent",
    route,
    {
        "respond": "respond",
        "lead": "lead"
    }
)

builder.add_edge("respond", END)
builder.add_edge("lead", END)

graph = builder.compile()


# ---------------- CHAT UI ---------------- #

for msg in st.session_state.state["chat"]:
    role = "🧑 User" if msg["role"] == "user" else "🤖 Bot"
    st.write(f"**{role}:** {msg['text']}")


user_input = st.chat_input("Type here...")

if user_input:

    state = st.session_state.state
    state["user_input"] = user_input

    state["chat"].append({"role": "user", "text": user_input})

    # run graph
    new_state = graph.invoke(state)

    st.session_state.state = new_state

    response = new_state.get("response", "")

    st.session_state.state["chat"].append({
        "role": "assistant",
        "text": response
    })

    st.rerun()