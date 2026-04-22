
# AutoStream AI Agent

## 1. Project Overview

AutoStream AI Agent is a conversational AI system designed for a fictional SaaS company that provides automated video editing tools for content creators. The agent acts as a sales and support assistant capable of handling user queries, retrieving product knowledge, and converting high-intent users into leads through a structured conversation flow.

The system combines:
- Intent classification
- Retrieval-Augmented Generation (RAG)
- Stateful conversation handling
- Tool execution for lead capture

---

## 2. How to Run the Project Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sahanaaacharyaa/autostream-ai-agent.git
cd autostream-ai-agent
````

### Step 2: Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### Step 5: Run the Application

```bash
streamlit run main.py
```

---

## 3. Architecture Explanation (≈200 words)

The system is built using a modular agent architecture powered by LangGraph. LangGraph was chosen over AutoGen because it provides better control over state transitions and deterministic workflow design, which is essential for structured lead collection flows.

The architecture consists of three main components:

1. **Intent Detection Node**
   The user input is classified into three categories: greeting, inquiry, or high-intent. This ensures that the conversation is routed correctly from the beginning.

2. **RAG Pipeline**
   A FAISS vector database stores embeddings of the product knowledge base, including pricing and policies. When a user asks a question, the system retrieves relevant context using HuggingFace embeddings and passes it to the LLM for grounded response generation. This prevents hallucinations and ensures factual accuracy.

3. **Lead Capture Tool Execution**
   When high intent is detected, the system transitions into a multi-step state machine that collects user details (name, email, platform, plan). LangGraph maintains state across steps, ensuring smooth conversation continuity. Once all required data is collected, a mock API function is triggered to simulate lead capture.

State is managed using a dictionary stored in Streamlit session state and passed through LangGraph nodes. This allows persistent memory across multiple turns while keeping the system lightweight and production-ready.

---

## 4. WhatsApp Deployment (Webhook Integration)

To deploy this agent on WhatsApp, the system can be integrated using WhatsApp Business API (or Twilio WhatsApp API).

### Proposed Flow:

1. A user sends a message on WhatsApp.
2. The message is received by a backend webhook (Flask/FastAPI server).
3. The webhook forwards the message to the LangGraph agent.
4. The agent processes:

   * Intent detection
   * RAG retrieval
   * Lead flow (if required)
5. The response is sent back to WhatsApp API.
6. User receives real-time reply.

### Architecture Components:

* WhatsApp Business API / Twilio
* Webhook Server (FastAPI/Flask)
* LangGraph AI Agent Backend
* Redis (optional for session tracking)

### Key Idea:

Each WhatsApp user is mapped to a unique session ID (phone number). This ensures persistent conversation state across multiple messages.

---

## 5. System Features

* Intent classification (greeting / inquiry / high-intent)
* RAG-based knowledge retrieval
* Stateful multi-turn conversation
* Lead qualification workflow
* Mock tool execution for CRM simulation

---

## 6. Tech Stack

* Python 3.9+
* Streamlit
* LangChain
* LangGraph
* FAISS
* HuggingFace Embeddings
* Groq LLM (Llama 3.1)

---

## 7. Knowledge Base

### Pricing

**Basic Plan**

* $29/month
* 10 videos/month
* 720p resolution

**Pro Plan**

* $79/month
* Unlimited videos
* 4K resolution
* AI captions

### Policies

* No refunds after 7 days
* 24/7 support only on Pro plan

---

## 8. Author

Built as part of an AI Agent assignment for AutoStream SaaS platform.

```

---

If you want, I can also give you next:
- :contentReference[oaicite:0]{index=0}
- `.gitignore`
- :contentReference[oaicite:1]{index=1}
- or :contentReference[oaicite:2]{index=2}

Just tell me.
```
