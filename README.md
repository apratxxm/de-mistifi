# Engineering Chat Extraction Pipeline

An LLM-powered tool to reconstruct structured project knowledge bases from raw chat history.

## Introduction
This project tackles the problem of "conversation-to-engineering memory extraction." Raw chat histories from long-running engineering projects contain invaluable context—why architectural decisions were made, how tricky bugs were debugged, and when project pivots occurred. However, simply asking an LLM to "summarize" a 20,000-line chat results in heavy compression loss and hallucinated timelines. 

This pipeline solves that by intelligently parsing the chat, grouping it into chronological sessions, and running multi-pass, targeted extractions (using a 120B parameter model) to pull out specific engineering decisions, debugging sequences, and project pivots without losing the surrounding context.

## Tech Used
- **Python 3.10+**
- **OpenAI Python SDK** (configured for OpenRouter)
- **OpenRouter API** (Using `openai/gpt-oss-120b:free` or `meta-llama/llama-3.3-70b-instruct:free`)
- **python-dotenv** (for secret management)

## Features
- **Smart Session Grouping**: Groups messages chronologically by conversation gaps to preserve causal reasoning (e.g., problem → debugging → fix) without artificially slicing context.
- **Overlapping Sub-Chunks**: Breaks large sessions into overlapping chunks (e.g., 80 turns with 15 turn overlap) so LLM context windows aren't overwhelmed and causal links aren't severed at chunk boundaries.
- **Multi-Pass Targeted Extraction**: Runs 6 separate, highly specific prompts per chunk to maximize recall:
  - ⏱️ `Timeline`: Every meaningful engineering action.
  - 🏛️ `Architecture`: System-level decisions and tradeoffs.
  - 🛠️ `Design`: Implementation-level choices and libraries.
  - 🐛 `Errors`: Step-by-step debugging sequences (hypothesis → experiment → result).
  - 🔄 `Pivots`: Changes in direction and their triggers.
  - 🧠 `Reasoning`: The engineering rationale, constraints, and rejected alternatives.
- **Granular Checkpointing**: Saves progress to disk after every single pass of every chunk. If the API rate limits or errors out, the script resumes exactly where it left off.
- **Automated Knowledge Base Generation**: Merges all extracted JSON data into a clean, searchable markdown document (`knowledge_base.md`).

## How to Run It

1. **Clone and setup virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install dependencies**
   ```powershell
   pip install openai python-dotenv
   ```

3. **Configure API Keys**
   - Get a free API key from [OpenRouter](https://openrouter.ai/keys).
   - Create a `.env` file in the root directory.
   - Add your key to the `.env` file:
     ```env
     OPENROUTER_API_KEY="sk-or-..."
     ```

4. **Prepare Input Data**
   - Ensure your raw chat export is saved as `DPR-Project-Intent-Document-Generation.txt` in the root directory (or update the `INPUT_FILE` variable in `extract.py`).

5. **Run Extraction**
   ```powershell
   python extract.py
   ```
   *Note: Checkpoints will automatically be saved to `extracted/checkpoints/`. If you need to stop the script, press `Ctrl+C`. Running it again will resume from the last saved checkpoint.*
