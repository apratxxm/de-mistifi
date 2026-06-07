"""
Internship Chat Extraction Pipeline
Parses DPR-Project-Intent-Document-Generation.txt and extracts structured knowledge
using OpenRouter API (free models).

Setup:
    pip install openai python-dotenv
    Add OPENROUTER_API_KEY=sk-or-... to .env in the same folder as this script
    Get free key at: https://openrouter.ai/keys
"""

import os
import re
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env from same directory as this script
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
INPUT_FILE = r"A:\pisha\hdfc internship extraction\DPR-Project-Intent-Document-Generation.txt"
OUTPUT_DIR = Path(r"A:\pisha\hdfc internship extraction\extracted")
SESSION_GAP_HOURS = 3    # Turns more than this apart = new session
CHUNK_SIZE_TURNS = 80    # Max turns per chunk fed to LLM
CHUNK_OVERLAP = 15       # Overlap between chunks to preserve context across boundaries
REQUESTS_PER_MINUTE = 18 # Free tier allows ~20 RPM
MODEL = "openai/gpt-oss-120b:free"  # Free model on OpenRouter (fallback: meta-llama/llama-3.3-70b-instruct:free)

# ─── SETUP ────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

if not API_KEY:
    print("ERROR: Set OPENROUTER_API_KEY in .env")
    print("  Get free key at: https://openrouter.ai/keys")
    exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# ─── STEP 1: PARSE ────────────────────────────────────────────────────────────

def parse_chat(filepath: str) -> list[dict]:
    """Parse the .txt into a list of turns: {speaker, timestamp, content}"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    turns = []
    parts = re.split(r'\n(gemini response|you asked)\n', raw)

    i = 1
    while i < len(parts) - 1:
        speaker_raw = parts[i].strip()
        content = parts[i + 1].strip()
        i += 2

        speaker = "gemini" if speaker_raw == "gemini response" else "user"

        ts_match = re.search(r'message time:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
        timestamp = None
        if ts_match:
            timestamp = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            content = re.sub(r'message time:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\n?', '', content).strip()

        if content:
            turns.append({"speaker": speaker, "timestamp": timestamp, "content": content})

    print(f"[Parser] {len(turns)} turns extracted")
    return turns


# ─── STEP 2: SESSION GROUPING ─────────────────────────────────────────────────

def group_into_sessions(turns: list[dict], gap_hours: int = 3) -> list[list[dict]]:
    sessions, current = [], []
    last_ts = None

    for turn in turns:
        ts = turn["timestamp"]
        if ts and last_ts and (ts - last_ts) > timedelta(hours=gap_hours):
            if current:
                sessions.append(current)
            current = []
        current.append(turn)
        if ts:
            last_ts = ts

    if current:
        sessions.append(current)

    print(f"[Sessioner] {len(sessions)} sessions")
    return sessions


# ─── STEP 3: CHUNK SESSIONS ───────────────────────────────────────────────────

def chunk_session(session: list[dict], size: int, overlap: int) -> list[list[dict]]:
    """Split a session into overlapping chunks of `size` turns."""
    if len(session) <= size:
        return [session]
    chunks = []
    start = 0
    while start < len(session):
        end = min(start + size, len(session))
        chunks.append(session[start:end])
        if end == len(session):
            break
        start += size - overlap
    return chunks


def format_chunk(chunk: list[dict]) -> str:
    lines = []
    for turn in chunk:
        ts_str = turn["timestamp"].strftime("%Y-%m-%d %H:%M") if turn["timestamp"] else "??:??"
        label = "USER" if turn["speaker"] == "user" else "GEMINI"
        lines.append(f"[{ts_str}] {label}:\n{turn['content']}\n")
    return "\n".join(lines)


# ─── STEP 4: EXTRACTION PROMPTS ───────────────────────────────────────────────

PASSES = {
    "timeline": """
Extract ALL engineering actions from this conversation chronologically.
Include small experiments, parameter changes, failed attempts, temporary fixes,
refactors, tool changes, and debugging steps. Prefer over-inclusion over omission.

For each action return:
- "timestamp": date/time string from conversation
- "event": specific action taken (not vague — be technical)
- "reason": why this action was taken
- "outcome": what happened as a result
- "blocking_issue": any blocker (null if none)

Return a JSON array only. No explanation, no prose.

CONVERSATION:
{chat}
""",

    "architecture": """
Extract every architectural decision (system-level choices: model selection,
database, pipeline design, infrastructure, retrieval strategy, data flow).

For each decision return:
- "decision": what was decided
- "reasoning": why (what constraints or failures led here)
- "alternatives_considered": list of rejected options
- "tradeoff": what was sacrificed
- "impact": consequence on the system
- "confidence": "high" | "medium" | "low" (how certain is this from the text)
- "evidence_quote": short verbatim quote showing this decision
- "caused_by": list of prior events/decisions that led to this
- "led_to": list of subsequent events/decisions this caused

Return a JSON array only. No prose.

CONVERSATION:
{chat}
""",

    "design": """
Extract every implementation-level design decision (library choice, data structure,
API design, file format, function structure, config values, hyperparameters).

For each return:
- "decision": what was chosen
- "context": what problem prompted this
- "reasoning": why this over alternatives
- "alternatives": what else was considered
- "outcome": did it work?
- "confidence": "high" | "medium" | "low"
- "caused_by": what led to this decision

Return a JSON array only. No prose.

CONVERSATION:
{chat}
""",

    "errors": """
Extract every debugging and error event. Include failed experiments, partial fixes,
and unresolved issues — not just fully solved bugs.

For each return:
- "problem": description of the issue
- "error_message": exact error text if mentioned (null if not)
- "logs": relevant log output (null if not present)
- "debugging_sequence": ordered list of {{hypothesis, experiment, result}} objects
  showing each step of the debugging process
- "root_cause": final identified cause (null if unresolved)
- "resolution": how it was fixed (null if unresolved)
- "lesson": key takeaway (null if none stated)

Return a JSON array only. No prose.

CONVERSATION:
{chat}
""",

    "pivots": """
Extract every moment where the project direction, approach, tool, or strategy changed.
Include small pivots (changing a parameter, switching a library) not just major ones.

For each pivot return:
- "before": original approach
- "after": new approach
- "trigger": event or finding that caused this
- "reasoning": why the old approach was abandoned
- "impact": consequence
- "confidence": "high" | "medium" | "low"

Return a JSON array only. No prose.

CONVERSATION:
{chat}
""",

    "reasoning": """
Extract the engineering reasoning and thinking process behind decisions.
Focus on WHY — constraints, assumptions, tradeoffs, uncertainties, rejected paths.

For each reasoning event return:
- "topic": what decision or action this applies to
- "constraint": limitation being worked around
- "assumption": what was assumed to be true
- "tradeoff": what was sacrificed
- "uncertainty": what was unknown at the time
- "rejected_alternatives": why other options were dropped
- "conclusion": final decision and rationale
- "confidence": "high" | "medium" | "low"

Return a JSON array only. No prose.

CONVERSATION:
{chat}
"""
}


# ─── STEP 5: API + CHECKPOINTING ──────────────────────────────────────────────

request_times: list[float] = []

def rate_limit():
    global request_times
    now = time.time()
    request_times = [t for t in request_times if now - t < 60]
    if len(request_times) >= REQUESTS_PER_MINUTE:
        sleep_for = 60 - (now - request_times[0]) + 1
        print(f"\n  [Rate limit] Sleeping {sleep_for:.1f}s...")
        time.sleep(sleep_for)


def call_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            rate_limit()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            request_times.append(time.time())
            return response.choices[0].message.content or "[]"
        except Exception as e:
            print(f"  [API Error attempt {attempt+1}] {e}")
            time.sleep(15)
    return "[]"


def parse_json_response(text: str) -> list:
    text = re.sub(r'^```(?:json)?\n?', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return []


def checkpoint_key(session_idx: int, chunk_idx: int, pass_name: str) -> Path:
    return CHECKPOINT_DIR / f"s{session_idx:03d}_c{chunk_idx:03d}_{pass_name}.json"


def load_checkpoint(session_idx: int, chunk_idx: int, pass_name: str) -> list | None:
    path = checkpoint_key(session_idx, chunk_idx, pass_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(session_idx: int, chunk_idx: int, pass_name: str, data: list):
    path = checkpoint_key(session_idx, chunk_idx, pass_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─── STEP 6: RUN EXTRACTION ───────────────────────────────────────────────────

def run_extraction(sessions: list[list[dict]]) -> dict:
    all_results = {p: [] for p in PASSES}

    for s_idx, session in enumerate(sessions):
        chunks = chunk_session(session, CHUNK_SIZE_TURNS, CHUNK_OVERLAP)
        session_date = next((t["timestamp"] for t in session if t["timestamp"]), f"session_{s_idx+1}")
        if isinstance(session_date, datetime):
            session_date = session_date.strftime("%Y-%m-%d")

        print(f"\n[Session {s_idx+1}/{len(sessions)}] {session_date} | {len(session)} turns → {len(chunks)} chunks")

        for c_idx, chunk in enumerate(chunks):
            chat_text = format_chunk(chunk)

            for pass_name, prompt_template in PASSES.items():
                # Check checkpoint first
                cached = load_checkpoint(s_idx, c_idx, pass_name)
                if cached is not None:
                    print(f"  [CACHED] s{s_idx+1} c{c_idx+1} {pass_name} ({len(cached)} items)")
                    for item in cached:
                        all_results[pass_name].append(item)
                    continue

                print(f"  [s{s_idx+1} c{c_idx+1}/{len(chunks)}] {pass_name}...", end=" ", flush=True)

                prompt = prompt_template.format(chat=chat_text)
                raw = call_gemini(prompt)
                extracted = parse_json_response(raw)

                # Tag with source metadata
                for item in extracted:
                    item["_session"] = s_idx + 1
                    item["_session_date"] = session_date
                    item["_chunk"] = c_idx + 1

                save_checkpoint(s_idx, c_idx, pass_name, extracted)
                all_results[pass_name].extend(extracted)
                print(f"→ {len(extracted)} items")

    return all_results


# ─── STEP 7: SAVE OUTPUTS ─────────────────────────────────────────────────────

def save_outputs(results: dict):
    for pass_name, items in results.items():
        out = OUTPUT_DIR / f"{pass_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, default=str)
        print(f"[Saved] {out} ({len(items)} items)")

    combined = OUTPUT_DIR / "all_extracted.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Saved] {combined}")

    md = OUTPUT_DIR / "knowledge_base.md"
    sections = {
        "timeline": "## Timeline",
        "architecture": "## Architectural Decisions",
        "design": "## Design Decisions",
        "errors": "## Errors & Debugging",
        "pivots": "## Pivot Points",
        "reasoning": "## Reasoning & Rationale"
    }
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Internship Project Knowledge Base\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for pass_name, items in results.items():
            f.write(f"\n{sections.get(pass_name, pass_name)}\n\n")
            for item in items:
                f.write("---\n\n")
                for k, v in item.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, list):
                        f.write(f"**{k}:**\n")
                        for vi in v:
                            if isinstance(vi, dict):
                                for dk, dv in vi.items():
                                    f.write(f"  - *{dk}*: {dv}\n")
                            else:
                                f.write(f"- {vi}\n")
                    else:
                        f.write(f"**{k}:** {v}\n")
                f.write("\n")
    print(f"[Saved] {md}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Internship Chat Extraction Pipeline ===\n")

    print(f"[1/4] Parsing {INPUT_FILE}...")
    turns = parse_chat(INPUT_FILE)

    print(f"\n[2/4] Grouping into sessions (gap={SESSION_GAP_HOURS}h)...")
    sessions = group_into_sessions(turns, gap_hours=SESSION_GAP_HOURS)

    total_chunks = sum(len(chunk_session(s, CHUNK_SIZE_TURNS, CHUNK_OVERLAP)) for s in sessions)
    total_calls = total_chunks * len(PASSES)
    est_minutes = total_calls / REQUESTS_PER_MINUTE
    print(f"\n[3/4] Running extraction...")
    print(f"      Sessions: {len(sessions)} | Chunks: ~{total_chunks} | API calls: ~{total_calls}")
    print(f"      Estimated time: ~{est_minutes:.0f} min (rate limited to {REQUESTS_PER_MINUTE} RPM)")
    print(f"      Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"      (Safe to interrupt and resume — checkpoints preserve progress)\n")

    results = run_extraction(sessions)

    print(f"\n[4/4] Saving outputs to {OUTPUT_DIR}...")
    save_outputs(results)

    print("\n=== Done ===")
    total_items = sum(len(v) for v in results.values())
    print(f"Total extracted items: {total_items}")
    for k, v in results.items():
        print(f"  {k}: {len(v)}")
