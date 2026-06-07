# Engineering Chat Extraction Pipeline

Extracts structured project knowledge bases from raw chat history.

## Introduction
Raw chat histories contain valuable engineering context (decisions, debugging, pivots) that gets lost when naive summarization is applied. This pipeline parses chat logs into sessions and uses a 120B parameter LLM to run multi-pass extractions, building a chronological, searchable engineering knowledge base without losing critical details.

## Tech Used
- **Python 3.10+**
- **OpenAI Python SDK** (OpenRouter)
- **OpenRouter API** (`openai/gpt-oss-120b:free`)
- **python-dotenv**

## Features
- **Session Grouping & Chunking**: Preserves causal chains without overwhelming LLM context windows.
- **Multi-Pass Extraction**: 6 targeted passes per chunk (Timeline, Architecture, Design, Errors, Pivots, Reasoning).
- **Auto-Checkpointing**: Saves progress per-pass to disk; instantly resumes if interrupted.
- **Markdown Export**: Auto-generates a clean `knowledge_base.md` from the extracted data.

## How to Run It

```powershell
# 1. Setup
python -m venv venv
.\venv\Scripts\activate
pip install openai python-dotenv

# 2. Add API key (from openrouter.ai/keys) to .env file
echo 'OPENROUTER_API_KEY="sk-or-..."' > .env

# 3. Run
python extract.py
```
