AutoStream AI Agent
Overview

AutoStream AI Agent is a conversational AI system designed for a fictional SaaS company that provides automated video editing tools for content creators. The agent is capable of handling user queries, retrieving contextual information using RAG, detecting user intent, and executing a structured lead capture workflow.

The system is built using Streamlit for the interface, LangGraph for state management, LangChain for orchestration, FAISS for vector search, and Groq LLM (Llama 3.1 8B) for reasoning and response generation.

Key Features
1. Intent Detection

The agent classifies user input into three categories:

Greeting (e.g., hi, hello)
Inquiry (pricing, features, policies)
High-intent lead (purchase intent like “I want Pro plan”)

A hybrid approach is used:

Rule-based keyword detection for fast routing
LLM fallback classification for robustness
2. RAG (Retrieval-Augmented Generation)

The agent uses a local knowledge base stored in knowledge_base.json, which includes:

Pricing Information

Basic Plan: $29/month, 10 videos/month, 720p resolution
Pro Plan: $79/month, unlimited videos, 4K resolution, AI captions

Policies

No refunds after 7 days
24/7 support available only for Pro plan

The data is embedded using sentence-transformers/all-MiniLM-L6-v2 and stored in a FAISS vector database for semantic retrieval.

This ensures:

Accurate responses
No hallucination
Context-aware answers
3. Lead Capture System (Tool Execution)

When high intent is detected, the agent triggers a structured lead collection flow:

It collects:

Name
Email
Platform (YouTube / Instagram / TikTok)
Plan (Basic / Pro)

Only after all required fields are collected, the system executes:

mock_lead_capture(name, email, platform, plan)

This simulates a backend CRM API call.

4. State Management (LangGraph)

The agent uses LangGraph to maintain conversation state across multiple turns.

States handled:
intent detection
response generation
lead collection flow
Lead flow steps:
Ask name
Ask email
Ask platform
Ask plan
Trigger tool execution

This ensures strict control over multi-step conversations and prevents premature API calls.

5. Memory Handling

The system maintains short-term memory of the last 5 chat interactions, allowing:

Context-aware responses
Continuity in conversation
Better user experience
Architecture
Components
Streamlit → UI layer
LangGraph → Conversation state machine
LangChain → Prompt + LLM orchestration
FAISS → Vector similarity search
HuggingFace Embeddings → Text vectorization
Groq LLM → Response generation
Flow
User input is received
Intent is detected
Router decides:
Respond (RAG-based answer)
Lead flow (multi-step collection)
If lead flow:
Collect user details step-by-step
Validate inputs
Execute mock API
Response is displayed in Streamlit UI
How to Run the Project
1. Clone the repository
git clone https://github.com/your-username/autostream-ai-agent.git
cd autostream-ai-agent
2. Install dependencies
pip install -r requirements.txt
3. Set environment variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
4. Run the application
streamlit run main.py
Requirements

Example requirements.txt:

streamlit
langchain
langgraph
langchain-groq
langchain-community
faiss-cpu
sentence-transformers
python-dotenv


The agent can be extended to WhatsApp using:

Option 1: Twilio WhatsApp API
Webhook receives WhatsApp messages
Messages forwarded to LangGraph agent
Response sent back via API
Option 2: Meta WhatsApp Cloud API
Use webhook endpoint for message ingestion
Connect backend to LangGraph pipeline
Flow:

User (WhatsApp) → Webhook → AI Agent → Response → WhatsApp

Why LangGraph?

LangGraph is used because it provides:

Strong state management for multi-turn flows
Deterministic routing logic
Easy handling of structured workflows like lead funnels
Production-ready conversation control
Future Improvements
Persistent database for leads (MongoDB/PostgreSQL)
Authentication system
Dashboard for analytics
WhatsApp / Telegram integration
Streaming responses for better UX
Conclusion

This project demonstrates a production-style conversational AI system combining:

RAG-based knowledge grounding
Intent-driven conversation flow
Stateful multi-step tool execution
Real-world SaaS use case simulation

It is designed to be scalable, modular, and deployable in real business environments.