import streamlit as st
import os
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END


# ---------------- ENV ---------------- #
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# ---------------- TOOL ---------------- #
def mock_lead_capture(name, email, platform, plan):
    print(f"\nLEAD CAPTURED: {name}, {email}, {platform}, {plan}\n")


# ---------------- KNOWLEDGE BASE ---------------- #
with open("data/knowledge_base.json") as f:
    KB = json.load(f)


def build_documents():
    docs = []
    for plan, details in KB["pricing"].items():
        docs.append(Document(page_content=f"{plan} plan: {details}"))
    for k, v in KB["policies"].items():
        docs.append(Document(page_content=f"{k}: {v}"))
    return docs


@st.cache_resource
def load_vectorstore():
    docs = build_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


llm = load_llm()


# ---------------- HELPERS ---------------- #
def is_email(text):
    return bool(re.fullmatch(r"[^@]+@[^@]+\.[^@]+", text))


def is_platform(text):
    text = text.lower()
    return any(p in text for p in ["youtube", "instagram", "tiktok", "linkedin"])


def is_plan(text):
    text = text.lower()
    return any(p in text for p in ["basic", "pro"])


# ---------------- INTENT ---------------- #
intent_prompt = PromptTemplate.from_template("""
Classify intent:
- greeting
- inquiry
- high_intent

User: {input}
Return only one word.
""")


def detect_intent(state):
    text = state["user_input"].lower()

    # 🔒 BLOCK if already in lead flow
    if state.get("collecting_lead"):
        return state

    if any(x in text for x in ["buy", "subscribe", "purchase", "get started", "i want"]):
        return {
            **state,
            "intent": "high_intent",
            "collecting_lead": True,
            "lead_step": "ask_name",
            "name": None,
            "email": None,
            "platform": None,
            "plan": None,
            "response": "Great! What’s your name?"
        }

    if any(x in text for x in ["hi", "hello", "hey"]):
        return {
            **state,
            "intent": "greeting",
            "response": "Hello! How can I help you today?"
        }

    if any(x in text for x in ["price", "plan", "cost", "feature"]):
        return {**state, "intent": "inquiry"}

    return {**state, "intent": "inquiry"}

# ---------------- RAG ---------------- #
def retrieve(query):
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])


def respond(state):
    context = retrieve(state["user_input"])

    history = "\n".join(
        [f"{m['role']}: {m['text']}" for m in state.get("chat", [])[-5:]]
    )

    prompt = f"""
You are AutoStream AI assistant.

Rules:
- Use ONLY context
- Be concise
- No hallucination

Chat:
{history}

Context:
{context}

User:
{state["user_input"]}
"""

    res = llm.invoke(prompt).content

    return {**state, "response": res}


# ---------------- LEAD FLOW ---------------- #
def lead_flow(state):

    step = state.get("lead_step")

    # STEP 1: NAME (ONLY NAME ALLOWED)
    if step == "ask_name":
        name = state["user_input"].strip()

        # prevent junk like "i want to buy"
        if len(name.split()) > 4:
            return {
                **state,
                "response": "Please enter a valid name."
            }

        state["name"] = name
        state["lead_step"] = "ask_email"
        return {**state, "response": "Great! What’s your email?"}

    # STEP 2: EMAIL
    if step == "ask_email":
        if is_email(state["user_input"]):
            state["email"] = state["user_input"]
            state["lead_step"] = "ask_platform"
            return {**state, "response": "Which platform? (YouTube / Instagram / TikTok)"}

        return {**state, "response": "Please enter a valid email."}

    # STEP 3: PLATFORM
    if step == "ask_platform":
        if is_platform(state["user_input"]):
            state["platform"] = state["user_input"].lower().strip()
            state["lead_step"] = "ask_plan"
            return {**state, "response": "Which plan? (Basic / Pro)"}

        return {**state, "response": "Choose YouTube / Instagram / TikTok"}

    # STEP 4: PLAN (STRICT)
    if step == "ask_plan":
        text = state["user_input"].lower().strip()

        if text not in ["basic", "pro"]:
            return {**state, "response": "❌ Choose ONLY Basic or Pro"}

        state["plan"] = text.title()

        mock_lead_capture(
            state["name"],
            state["email"],
            state["platform"],
            state["plan"]
        )

        return {
            **state,
            "response": "🎉 Lead captured successfully! We’ll contact you soon.",
            "collecting_lead": False,
            "lead_step": None
        }

    return state

# ---------------- ROUTER ---------------- #
def route(state):

    # 🔒 HARD PRIORITY: lead always wins
    if state.get("collecting_lead") or state.get("lead_step"):
        return "lead"

    if state["intent"] in ["greeting", "inquiry"]:
        return "respond"

    if state["intent"] == "high_intent":
        return "lead"

    return "respond"


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


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="AutoStream AI Agent", layout="centered")

st.markdown("""
<style>
.stApp { background-color: transparent; }

.main-title {
    font-size: 28px;
    font-weight: 600;
    text-align: center;
}

.sub-title {
    font-size: 14px;
    text-align: center;
    opacity: 0.7;
}

.user-msg {
    max-width: 75%;
    padding: 10px;
    border-radius: 12px;
    background: rgba(0,0,0,0.04);
    margin-left: auto;
    margin: 6px 0;
}

.bot-msg {
    max-width: 75%;
    padding: 10px;
    border-radius: 12px;
    background: rgba(0,0,0,0.07);
    margin-right: auto;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>AutoStream AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI assistant for pricing, recommendations and lead capture</div>", unsafe_allow_html=True)

st.divider()


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
        "collecting_lead": False,
        "lead_step": None
    }


# ---------------- CHAT ---------------- #
for msg in st.session_state.state["chat"]:
    align = "flex-end" if msg["role"] == "user" else "flex-start"
    cls = "user-msg" if msg["role"] == "user" else "bot-msg"

    st.markdown(f"""
    <div style="display:flex; justify-content:{align};">
        <div class="{cls}">{msg['text']}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- INPUT ---------------- #
user_input = st.chat_input("Type your message...")

if user_input:

    state = st.session_state.state
    state["user_input"] = user_input

    state["chat"].append({"role": "user", "text": user_input})

    new_state = graph.invoke(state)

    # 🔒 SAFETY FALLBACK
    if "response" not in new_state or new_state["response"] is None:
        new_state["response"] = "I'm here to help. Could you rephrase?"

    st.session_state.state = new_state

    st.session_state.state["chat"].append({
        "role": "assistant",
        "text": new_state["response"]
    })

    st.rerun()