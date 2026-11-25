# AI Conflict Resolution Assistant

A multi-agent system that analyzes tense text conversations, maps emotional escalation, and suggests empathetic, boundary-respecting replies. It combines lightweight scikit-learn classifiers for line-level emotion and conflict detection with a CrewAI agent crew powered by Groq (`openai/gpt-oss-20b`).

## Features

- **Automatic classifier setup** — On first import, `ml_models.py` trains TF-IDF + Logistic Regression models on synthetic data and saves them under `models/` for reuse.
- **Emotional timeline tool** — Agents call `Analyze Emotional Timeline` to label each utterance with emotion and conflict risk before reasoning.
- **Sequential multi-agent report** — Emotional Analyst → Perspective Balancer → De-escalation Coach produce a structured resolution report.
- **Low-latency LLM inference** — Language model calls go through Groq (`groq/openai/gpt-oss-20b`) via LiteLLM.

## Architecture

The pipeline has two layers:

1. **Classification layer** — Synthetic-data classifiers predict emotion (e.g. Frustrated, Accusatory, Neutral, Empathetic, Defensive) and conflict type (e.g. Boundary violation, Expectation mismatch, Respect issue, Trust issue) per line.
2. **Agent layer** — A CrewAI sequential crew reasons over those labels and drafts response options.

```text
Conversation transcript
        │
        ▼
Analyze Emotional Timeline  ──►  TF-IDF + LogisticRegression
        │
        ▼
Emotional Analyst
        │
        ▼
Perspective Balancer
        │
        ▼
De-escalation Coach
        │
        ▼
Final Resolution Report
```

| Agent | Role in the run |
| --- | --- |
| Emotional Analyst | Uses the timeline tool; summarizes escalation points |
| Perspective Balancer | Explains what the other person may be feeling |
| De-escalation Coach | Synthesizes prior outputs into the final report |
| Boundary Guardian | Defined and registered in the crew, but **not assigned a task** in the current workflow |

## Project structure

```text
AI-Conflict-Resolution-Assistant/
├── main.py              # Entry point; sample conversation and crew kickoff
├── crew.py              # LLM config, agents, tasks, sequential crew
├── tools.py             # CrewAI tool wrapping the ML predictors
├── ml_models.py         # Synthetic data, train/load, emotion & conflict inference
├── requirements.txt     # Python dependencies
├── .env.example         # Template for Groq API key
└── models/              # Created at runtime (vectorizers & classifiers)
```

## Prerequisites

- Python **3.10–3.12** (Python 3.14 is not supported by this dependency stack)
- A [Groq](https://console.groq.com/) API key

## Installation

Use a project virtual environment so packages do not conflict with other Python installs:

```bash
# Prefer an explicit 3.10/3.11/3.12 interpreter when multiple are installed
# Windows example: py -3.10 -m venv .venv
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Always activate `.venv` before running or installing packages for this project.

## Configuration

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

If the key is missing, the crew still builds but LLM calls will fail; `crew.py` prints a warning when `GROQ_API_KEY` is unset.

## Usage

Run the sample conversation shipped in `main.py`:

```bash
python main.py
```

On the first run, classifiers are trained (if `models/` is empty) and then the crew analyzes the transcript.

### Analyze your own conversation

Replace the `sample_conversation` string in `main.py` with lines in `Speaker: utterance` form, for example:

```text
User: I can't believe you changed the report last minute without telling me.
Other: You always overreact, it's not a big deal.
```

Then run `python main.py` again.

## Example output

The De-escalation Coach is instructed to return a report with exactly these sections:

1. **Emotional Summary** — Who escalated and where
2. **Escalation Points** — Specific messages flagged as high risk
3. **Suggested Responses**
   - Option 1: De-escalation
   - Option 2: Boundary-respecting
   - Option 3: Empathy-forward
4. **Ethical Disclaimer** — Reminder that human judgment is required before sending any reply

## How it works

| File | Responsibility |
| --- | --- |
| `ml_models.py` | Generates synthetic training data, loads or trains models, exposes `predict_emotion` / `predict_conflict` |
| `tools.py` | Parses a multi-line transcript and wraps predictions as the `Analyze Emotional Timeline` tool |
| `crew.py` | Configures Groq LLM, agents, tasks, and `create_crew()` |
| `main.py` | Builds the crew and prints the final `CrewOutput` |

## Limitations

- Classifiers are trained on a **small synthetic dataset**; they illustrate the pipeline and are not production-grade NLP.
- This is **not** a substitute for professional mediation, counseling, or legal advice.
- The Boundary Guardian agent is present in code but does not currently contribute to the task chain.
- Always apply human judgment before sending AI-suggested messages.

## Contributing

Issues and pull requests are welcome. For larger changes, open an issue first so the approach can be discussed.