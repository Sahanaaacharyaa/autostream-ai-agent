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


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def mock_lead_capture(name, email, platform, plan):
    print(f"Lead captured successfully: {name}, {email}, {platform}, {plan}")


with open("data/knowledge_base.json") as f:
    KB = json.load(f)

def build_documents():
    docs = []
    for plan, details in KB["pricing"].items():
        docs.append(Document(page_content=f"{plan} plan: {details}"))
    for key, value in KB["policies"].items():
        docs.append(Document(page_content=f"{key}: {value}"))
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


def is_email(text):
    return re.match(r"[^@]+@[^@]+\.[^@]+", text)

def is_platform(text):
    text = text.lower()
    return any(p in text for p in ["youtube", "instagram", "tiktok", "linkedin"])

def is_plan(text):
    text = text.lower()
    return any(p in text for p in ["basic", "pro"])


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

  
    if "change my mind" in text or "other one" in text or "exit" in text:
        return {
            **state,
            "intent": "greeting",
            "collecting_lead": False
        }

    if any(x in text for x in [
        "buy", "subscribe", "purchase", "sign up",
        "i want", "sounds good", "let's do it", "get started"
    ]):
        return {**state, "intent": "high_intent"}

    if any(x in text for x in [
        "price", "plan", "cost", "feature", "pricing", "subscription"
    ]):
        return {**state, "intent": "inquiry"}

    res = llm.invoke(intent_prompt.format(input=state["user_input"]))
    intent = res.content.strip().lower()

    if intent not in ["greeting", "inquiry", "high_intent"]:
        intent = "inquiry"

    return {**state, "intent": intent}


def retrieve_info(query):
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

def generate_response(state):
    context = retrieve_info(state["user_input"])

    if not context.strip():
        return {**state, "response": "I couldn't find that info. Can you rephrase?"}

    history = "\n".join([
        f"{m['role']}: {m['text']}"
        for m in st.session_state.chat[-5:]
    ])

    prompt = f"""
You are AutoStream AI assistant.

Rules:
- Answer ONLY from context
- Be clear and structured
- Do not hallucinate

Conversation:
{history}

Context:
{context}

User:
{state['user_input']}
"""

    res = llm.invoke(prompt)
    return {**state, "response": res.content}


def lead_collection(state):

    state["collecting_lead"] = True

    if not state.get("name"):
        return {**state, "response": "Great! Let’s get started — what’s your name?"}

    if not state.get("email"):
        if is_email(state["user_input"]):
            state["email"] = state["user_input"]
        else:
            return {**state, "response": "Please enter a valid email (example: name@gmail.com)"}

    if not state.get("platform"):
        if is_platform(state["user_input"]):
            state["platform"] = state["user_input"]
        else:
            return {**state, "response": "Which platform? (YouTube / Instagram / TikTok)"}

    if not state.get("plan"):
        if is_plan(state["user_input"]):
            state["plan"] = state["user_input"].title()
        else:
            return {**state, "response": "Which plan do you want? (Basic / Pro)"}

   
    mock_lead_capture(
        state["name"],
        state["email"],
        state["platform"],
        state["plan"]
    )

    st.success(f"LEAD CAPTURED: {state['name']} | {state['email']} | {state['platform']} | {state['plan']}")

    return {
        "user_input": "",
        "intent": None,
        "response": "Done! We’ll contact you soon ",
        "name": None,
        "email": None,
        "platform": None,
        "plan": None,
        "collecting_lead": False
    }


def route(state):
    if state.get("collecting_lead"):
        return "lead"
    if state["intent"] == "high_intent":
        return "lead"
    return "respond"


builder = StateGraph(dict)

builder.add_node("intent", detect_intent)
builder.add_node("respond", generate_response)
builder.add_node("lead", lead_collection)

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


st.set_page_config(
    page_title="AutoStream AI Agent",
    layout="centered"
)


st.markdown("""
    <style>

    /* Global background */
    .stApp {
        background-color: transparent;
    }

    /* Title */
    .main-title {
        font-size: 28px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 5px;
        color: inherit;
    }

    .sub-title {
        font-size: 14px;
        text-align: center;
        margin-bottom: 20px;
        opacity: 0.7;
    }

    /* Chat bubbles */
    .user-msg {
        display: inline-block;
        max-width: 75%;
        width: fit-content;
        padding: 10px 12px;
        margin: 6px 0;
        border-radius: 12px;
        border: 0px solid rgba(0,0,0,0.2);
        background: rgba(0,0,0,0.04);

        /* align right */
        margin-left: auto;
        text-align: left;
        word-wrap: break-word;
    }

    .bot-msg {
        display: inline-block;
        max-width: 75%;
        width: fit-content;
        padding: 10px 12px;
        margin: 6px 0;
        border-radius: 12px;
        border: 0px solid rgba(0,0,0,0.2);
        background: rgba(0,0,0,0.07);

        /* align left */
        margin-right: auto;
        text-align: left;
        word-wrap: break-word;
    }

    /* Sidebar clean */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(0,0,0,0.1);
    }

    /* Input box spacing */
    .stChatInputContainer {
        border-top: 1px solid rgba(0,0,0,0.1);
        padding-top: 10px;
    }

    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-title'>AutoStream AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI assistant for pricing, recommendations and lead capture</div>", unsafe_allow_html=True)

st.divider()


with st.sidebar:
    st.header("Plans Overview")

    st.subheader("Basic Plan")
    st.write("• 10 videos per month")
    st.write("• 720p resolution")
    st.write("• Standard features")

    st.markdown("---")

    st.subheader("Pro Plan")
    st.write("• Unlimited videos")
    st.write("• 4K resolution")
    st.write("• AI captions")
    st.write("• 24/7 support")

    st.markdown("---")
    st.write("Tip: Try asking comparison or pricing questions")


if "state" not in st.session_state:
    st.session_state.state = {
        "user_input": "",
        "intent": None,
        "response": None,
        "name": None,
        "email": None,
        "platform": None,
        "plan": None,
        "collecting_lead": False
    }

if "chat" not in st.session_state:
    st.session_state.chat = []


for msg in st.session_state.chat:

    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end;">
            <div class="user-msg">
                {msg['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-start;">
            <div class="bot-msg">
                {msg['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

user_input = st.chat_input("Type your message here...")

if user_input:

    st.session_state.chat.append({"role": "user", "text": user_input})

    state = st.session_state.state
    state["user_input"] = user_input

 
    if state.get("collecting_lead"):
        if not state.get("name"):
            state["name"] = user_input

        elif not state.get("email") and is_email(user_input):
            state["email"] = user_input

        elif not state.get("platform") and is_platform(user_input):
            state["platform"] = user_input

        elif not state.get("plan") and is_plan(user_input):
            state["plan"] = user_input.title()

    state = graph.invoke(state)
    st.session_state.state = state

    response = state["response"]

    st.session_state.chat.append({"role": "assistant", "text": response})

    st.rerun()