import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from rag import build_retriever
from tools import mock_lead_capture


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
retriever = build_retriever()


# ---------------- INTENT DETECTION ---------------- #

def detect_intent(state):
    text = state["user_input"].lower()

    if any(x in text for x in ["hi", "hello", "hey"]):
        return {**state, "intent": "greeting"}

    if any(x in text for x in ["price", "plan", "cost", "feature"]):
        return {**state, "intent": "inquiry"}

    if any(x in text for x in [
        "buy", "subscribe", "sign up", "i want pro", "get started", "sounds good"
    ]):
        return {
            **state,
            "intent": "high_intent",
            "collecting_lead": True
        }

    prompt = PromptTemplate.from_template(
        "Classify intent: greeting, inquiry, high_intent\nUser: {input}\nReturn one word."
    )

    res = llm.invoke(prompt.format(input=text)).content.lower().strip()

    if res not in ["greeting", "inquiry", "high_intent"]:
        res = "inquiry"

    return {**state, "intent": res}


# ---------------- RAG RESPONSE ---------------- #

def respond(state):
    docs = retriever.invoke(state["user_input"])
    context = "\n".join([d.page_content for d in docs])

    history = "\n".join(
        [f"{m['role']}: {m['text']}" for m in state.get("chat", [])[-5:]]
    )

    prompt = f"""
You are AutoStream AI assistant.

Rules:
- Use ONLY provided context
- Be concise and helpful

Chat History:
{history}

Context:
{context}

User:
{state["user_input"]}
"""

    response = llm.invoke(prompt).content

    return {
        **state,
        "response": response
    }


# ---------------- LEAD FLOW ---------------- #

def is_email(x):
    return re.match(r"[^@]+@[^@]+\.[^@]+", x)


def lead_flow(state):
    state["collecting_lead"] = True

    if not state.get("name"):
        return {**state, "response": "Great! What’s your name?"}

    if not state.get("email"):
        if is_email(state["user_input"]):
            state["email"] = state["user_input"]
        else:
            return {**state, "response": "Please enter a valid email."}

    if not state.get("platform"):
        state["platform"] = state["user_input"]
        return {**state, "response": "Which plan? (Basic / Pro)"}

    if not state.get("plan"):
        state["plan"] = state["user_input"]

    mock_lead_capture(
        state["name"],
        state["email"],
        state["platform"],
        state["plan"]
    )

    return {
        **state,
        "response": "🎉 Lead captured successfully! We’ll contact you soon.",
        "collecting_lead": False
    }


# ---------------- ROUTER ---------------- #

def route(state):
    if state.get("collecting_lead"):
        return "lead"

    if state["intent"] == "high_intent":
        return "lead"

    return "respond"