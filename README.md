

# AutoStream AI Agent

## Overview

AutoStream AI Agent is a conversational AI system built for a fictional SaaS product that provides automated video editing tools for content creators.

The agent can:

* Understand user intent (greeting, inquiry, high-intent purchase)
* Answer questions using a RAG-based knowledge base
* Recommend pricing plans and features
* Collect leads through a structured conversation flow
* Store captured lead information via a mock backend function

---

## Features

### 1. Intent Detection

The system classifies user input into:

* Greeting (e.g., "hi", "hello")
* Inquiry (pricing, features, plans)
* High-intent users (ready to purchase or sign up)

A combination of rule-based logic and LLM-based classification is used for better accuracy.

---

### 2. RAG (Retrieval-Augmented Generation)

The agent retrieves contextual knowledge from a local FAISS vector database built using:

* Pricing information
* Feature descriptions
* Company policies

This ensures responses are grounded and accurate.

---

### 3. Lead Capture System

When a user shows purchase intent, the agent collects:

* Name
* Email
* Platform (YouTube / Instagram / TikTok)
* Plan (Basic / Pro)

After collecting all details, a mock function simulates backend lead storage.

---

### 4. Stateful Conversation

The system maintains conversation memory using Streamlit session state and LangGraph state management, enabling multi-turn interactions.

---

## Tech Stack

* Python 3.9+
* Streamlit (UI)
* LangChain (LLM orchestration)
* LangGraph (state machine workflow)
* FAISS (vector database)
* HuggingFace Embeddings
* Groq LLM (LLaMA 3.1 8B Instant)
* dotenv (environment variables)

---

## Project Structure

```
autostream-agent/
│── main.py
│── data/
│    └── knowledge_base.json
│── requirements.txt
│── .env
│── README.md
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com//Sahanaaacharyaa.git
cd autostream-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the app

```bash
streamlit run main.py
```

---

## Architecture Explanation (≈200 words)

The AutoStream AI Agent is built using a modular architecture combining LangChain, LangGraph, and Streamlit. The system follows a state-driven workflow where each user interaction updates a shared state object.

LangGraph is used to define a controlled flow between three main nodes:

1. Intent detection
2. Response generation (RAG-based)
3. Lead collection

This ensures deterministic behavior for lead qualification while still allowing flexible LLM responses for general queries.

A Retrieval-Augmented Generation (RAG) pipeline is implemented using FAISS vector storage and HuggingFace embeddings. The knowledge base (pricing and policies) is embedded locally, allowing the LLM to retrieve relevant context before generating responses.

State management is handled using Streamlit session state combined with LangGraph state transitions. This enables multi-turn conversations where the system remembers user details such as name, email, platform, and selected plan.

The design ensures a balance between structured business logic (lead capture flow) and natural language understanding (LLM responses). This makes the system production-oriented and easily extensible for CRM integration or WhatsApp deployment.

---

## WhatsApp Integration (Deployment Idea)

To integrate this agent with WhatsApp:

1. Use **WhatsApp Business API** or **Twilio WhatsApp API**
2. Create a webhook server using **FastAPI or Flask**
3. Forward incoming WhatsApp messages to the LangGraph agent
4. Maintain user session state using:

   * Redis or database (instead of Streamlit session state)
5. Send agent response back via WhatsApp API

Flow:
WhatsApp User → Webhook → AI Agent → Response → WhatsApp API

This allows the same logic to work as a real production chatbot.

---

## Deliverables Summary

This project includes:

* Intent detection system
* RAG-based knowledge retrieval
* Stateful conversation engine
* Lead capture workflow
* Mock CRM integration
* Streamlit UI for demonstration

---

## Notes

* This project is designed for demonstration and evaluation purposes.
* The lead capture function can be replaced with real CRM or database integration.
* The architecture is scalable for production deployment.

---


