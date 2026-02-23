#  AI-Powered Conflict Resolution Assistant

A multi-agent AI system designed to analyze tense text conversations, map emotional escalation, and suggest empathetic, boundary-respecting responses.

This project bridges the gap between automated text analysis (for fast emotion detection) and conversational AI (for reasoning and human-like advice), orchestrated via CrewAI.

##  Overview

When humans argue over text, it is easy to misinterpret tone and escalate the conflict. This tool acts as an objective, multi-perspective mediator. It reads a conversation transcript and uses a team of specialized AI agents to break down what went wrong and how to fix it.

### The Dual-Layer Architecture

1. **The Classification Layer:** A custom machine learning pipeline automatically trains itself on synthetic data to classify *Emotions* (e.g., Frustrated, Accusatory) and *Conflict Risk* (e.g., Boundary Violation, Expectation Mismatch) line-by-line.
2. **The Agent Layer:** A CrewAI setup powered by Groq (Llama 3.1) takes those structural classifications and reasons over them to generate psychological insights and actionable de-escalation strategies.

##  Key Features

* **Automated Setup:** The `ml_models.py` script requires no manual configuration. If it doesn't find existing model files on your machine, it automatically generates a baseline dataset, trains the classifiers, and saves them for future use.
* **Custom Agent Tools:** The classification models are wrapped into a callable tool (`tools.py`), giving the AI agents the ability to instantly detect the emotional tone and conflict risk of a conversation before formulating their advice.
* **Multi-Agent Orchestration:** Utilizes four distinct personas (Analyst, Balancer, Guardian, and Coach) to ensure the final advice is balanced, safe, and empathetic.
* **Lightning Fast Inference:** Uses Groq's ultra-low latency infrastructure to power the language models.

##  Repository Structure

* `ml_models.py`: The core classification engine. Handles synthetic data generation, model training, and saving the trained files locally.
* `tools.py`: The bridge file. Parses raw conversations and packages the classification models into an "Analyze Emotional Timeline" tool for the agents to use.
* `crew.py`: Defines the language model configuration, the four specific Agent personas, their designated Tasks, and orchestrates the sequential workflow.
* `main.py`: The entry point. Contains a sample argument and kicks off the multi-agent analysis process.

##  Getting Started

### 1. Prerequisites

You will need Python installed along with the following packages:

```bash
pip install scikit-learn crewai python-dotenv

```

### 2. Environment Setup

This project uses Groq to power the language models. Create a `.env` file in the root of your project and add your API key securely:

```bash
# .env
GROQ_API_KEY=your_actual_api_key_here

```

### 3. Run the Assistant

Simply execute the main script. On the first run, it will automatically train the classifiers, save them to a new `models/` directory, and then kick off the multi-agent analysis.

```bash
python main.py

```

##  Example Output

When you run `main.py`, the agents will analyze the provided sample conversation and output a structured resolution report containing:

1. **Emotional Summary:** A breakdown of who escalated the conversation and where.
2. **Escalation Points:** Specific lines flagged by the classification tool as highly accusatory or defensive.
3. **Suggested Responses:** Three distinct ways the user can reply (De-escalation, Boundary-respecting, and Empathy-forward).
4. **Ethical Disclaimer:** A reminder that human judgment is required before sending AI-generated responses.
