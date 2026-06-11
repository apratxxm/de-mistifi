import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# Setup paths
BASE_DIR = r"a:\pisha\hdfc internship extraction\extracted"
INPUT_FILE = os.path.join(BASE_DIR, "all_extracted.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "INTERNSHIP_MASTER_DOCUMENT.md")

PHASE_KEYWORDS = {
    "Data Preparation": [r"\bexcel\b", r"\bquery\b", r"\bintent\b", r"\bmapping\b", r"\bclean\b"],
    "Passage Engineering": [r"\bkeybert\b", r"\bkeyword\b", r"\bkmeans\b", r"\bcluster\b", r"\bdraft\b"],
    "Zero-Shot DPR": [r"\bzero-shot\b", r"\bretrieval\b", r"\bpassage\b", r"\bdpr\b", r"\bbaseline\b"],
    "Active Learning": [r"\bactive learning\b", r"\bmisclassification\b", r"\brelabel\b", r"\bconfidence\b"],
    "Training V1": [r"\btripletloss\b", r"\btripletevaluator\b", r"\bv1\b", r"\boverfit\b"],
    "Training V2": [r"\bmultiplenegativesrankingloss\b", r"\binformationretrievalevaluator\b", r"\bv2\b"],
    "Training V3": [r"\boversampling\b", r"\bbatch size\b", r"\bv3\b", r"\bimbalance\b"],
    "Data Efficiency": [r"\b450 examples\b", r"\bv4\b", r"\befficiency\b", r"\bhalf data\b"]
}

def parse_ts(ts):
    if not ts or ts == "Unknown":
        return datetime.max
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            return datetime.strptime(ts, "%Y-%m-%d")
        except ValueError:
            return datetime.max

def extract_title(records):
    for r in records:
        for key in ['event', 'decision', 'problem', 'topic']:
            val = r['raw'].get(key)
            if val and isinstance(val, str):
                if len(val) > 100:
                    return val[:100] + "..."
                return val
    text = records[0]["text"].split(" | ")[0]
    return text[:100] + "..." if len(text) > 100 else text

def assign_phase(title, topic_text):
    title_lower = title.lower()
    text_lower = topic_text.lower()
    
    title_weight = 3
    scores = defaultdict(int)
    
    for phase, keywords in PHASE_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, title_lower):
                scores[phase] += title_weight
            if re.search(kw, text_lower):
                scores[phase] += 1
                
    if scores:
        return max(scores, key=scores.get)
    return "Other"

def extract_milestones(all_records):
    milestones = []
    metric_pattern = re.compile(r'(\d+(\.\d+)?)\s*%?')
    valid_keywords = ["top-1", "top1", "top-3", "accuracy", "baseline", "precision", "recall"]
    
    for r in all_records:
        text = str(r["raw"]).lower()
        if any(kw in text for kw in valid_keywords) and metric_pattern.search(text):
            for k, v in r["raw"].items():
                if isinstance(v, str):
                    v_lower = v.lower()
                    if any(kw in v_lower for kw in valid_keywords) and metric_pattern.search(v_lower):
                        if v not in [m["text"] for m in milestones]:
                            milestones.append({"text": v, "timestamp": r["timestamp"]})
    milestones.sort(key=lambda x: parse_ts(x["timestamp"]))
    return milestones

def extract_versions(all_records):
    versions = []
    version_pattern = re.compile(r'\bv\d+(\.\d+)?\b', re.IGNORECASE)
    for r in all_records:
        for k, v in r["raw"].items():
            if isinstance(v, str):
                match = version_pattern.search(v)
                if match:
                    ver = match.group(0).upper()
                    if not any(x["version"] == ver for x in versions):
                        versions.append({"version": ver, "text": v, "timestamp": r["timestamp"]})
    versions.sort(key=lambda x: parse_ts(x["timestamp"]))
    return versions

def extract_lessons(all_records):
    lessons = []
    for r in all_records:
        for key in ['lesson', 'tradeoff', 'constraint', 'uncertainty']:
            val = r['raw'].get(key)
            if val and isinstance(val, str):
                is_duplicate = False
                for existing in lessons:
                    similarity = fuzz.ratio(val.lower(), existing["text"].lower())
                    if similarity > 85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    lessons.append({
                        "type": key.capitalize(),
                        "text": val,
                        "context": r["text"]
                    })
    return lessons

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 1. Normalize
    print("Normalizing records...")
    core_records = []
    accessory_records = []
    all_records_flat = []
    
    for category, items in data.items():
        for idx, item in enumerate(items):
            text_parts = []
            for k, v in item.items():
                if v and isinstance(v, str) and k not in ["_session", "_session_date", "timestamp", "_chunk"]:
                    text_parts.append(f"{k}: {v}")
            
            text = " | ".join(text_parts)
            timestamp = item.get("timestamp") or item.get("date") or "Unknown"
            
            record = {
                "id": f"{category[:3].upper()}_{idx:04d}",
                "category": category,
                "text": text,
                "timestamp": timestamp,
                "raw": item
            }
            all_records_flat.append(record)
            
            if category in ["timeline", "architecture", "design"]:
                core_records.append(record)
            elif category in ["reasoning", "pivots", "errors"]:
                accessory_records.append(record)

    if not core_records:
        print("No core records found. Exiting.")
        return

    # 2. Extract Milestones & Lessons
    print("Extracting milestones, versions, and lessons...")
    milestones = extract_milestones(all_records_flat)
    versions = extract_versions(all_records_flat)
    lessons = extract_lessons(all_records_flat)

    # 3. Embeddings
    print("Loading model all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding core records...")
    core_texts = [r["text"] for r in core_records]
    core_embeddings = model.encode(core_texts, show_progress_bar=True)
    
    print("Encoding accessory records...")
    if accessory_records:
        acc_texts = [r["text"] for r in accessory_records]
        acc_embeddings = model.encode(acc_texts, show_progress_bar=True)

    # 4. Semantic Clustering of Core Records
    print("Clustering core records...")
    clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.18)
    core_labels = clustering.fit_predict(core_embeddings)

    topics = defaultdict(list)
    for i, label in enumerate(core_labels):
        topics[label].append(core_records[i])

    # 5. Attach Accessory Records via Similarity
    if accessory_records:
        print("Attaching accessory records to topics...")
        sim_matrix = cosine_similarity(acc_embeddings, core_embeddings)
        
        ATTACH_THRESHOLD = 0.42
        MAX_ATTACHMENTS = 2
        
        for i, acc_record in enumerate(accessory_records):
            sims = sim_matrix[i]
            best_idxs = np.argsort(sims)[::-1]
            attached = 0
            
            for idx in best_idxs:
                if sims[idx] < ATTACH_THRESHOLD:
                    break
                
                label = core_labels[idx]
                topics[label].append(acc_record)
                attached += 1
                
                if attached >= MAX_ATTACHMENTS:
                    break

    # 6. Build Final Concepts & Assign Phases
    concepts = []
    MAX_TOPIC_SIZE = 20
    for label, records in topics.items():
        records.sort(key=lambda x: parse_ts(x["timestamp"]))
        
        for i in range(0, len(records), MAX_TOPIC_SIZE):
            chunk = records[i:i + MAX_TOPIC_SIZE]
            title = extract_title(chunk)
            primary_ts = chunk[0]["timestamp"]
            
            # Combine text for phase assignment
            combined_text = " ".join([r["text"] for r in chunk])
            phase = assign_phase(title, combined_text)
            
            chunk_suffix = f"_pt{i//MAX_TOPIC_SIZE + 1}" if len(records) > MAX_TOPIC_SIZE else ""
            
            concepts.append({
                "concept_id": f"TOPIC_{label:04d}{chunk_suffix}",
                "title": title,
                "primary_timestamp": primary_ts,
                "phase": phase,
                "records": chunk,
                "sort_key": parse_ts(primary_ts)
            })

    # Group by Phase, then sort topics chronologically within each phase
    phases_dict = defaultdict(list)
    for c in concepts:
        phases_dict[c["phase"]].append(c)
        
    for p in phases_dict:
        phases_dict[p].sort(key=lambda x: x["sort_key"])

    # 7. Generate Markdown
    print("Generating Markdown...")
    
    total_topics = len(concepts)
    counts = defaultdict(int)
    for c in concepts:
        for r in c["records"]:
            counts[r["category"]] += 1
            
    # Define order of phases
    ordered_phases = list(PHASE_KEYWORDS.keys()) + ["Other"]

    md_lines = []
    md_lines.append("# Internship Master Document")
    md_lines.append("\n*Generated via Semantic Consolidation Pipeline*\n")
    
    # Table of Contents
    md_lines.append("## Table of Contents\n")
    toc_idx = 1
    md_lines.append(f"{toc_idx}. Executive Summary")
    toc_idx += 1
    if milestones:
        md_lines.append(f"{toc_idx}. Major Milestones")
        toc_idx += 1
    if versions:
        md_lines.append(f"{toc_idx}. Model Evolution")
        toc_idx += 1
    if lessons:
        md_lines.append(f"{toc_idx}. Engineering Learnings")
        toc_idx += 1
    for phase_name in ordered_phases:
        if phase_name in phases_dict and phases_dict[phase_name]:
            md_lines.append(f"{toc_idx}. Phase: {phase_name}")
            toc_idx += 1
    md_lines.append("\n---\n")
    
    # Executive Summary
    md_lines.append("## Executive Summary\n")
    md_lines.append(f"**Total Topics**: {total_topics}")
    md_lines.append(f"- Timeline Events: {counts.get('timeline', 0)}")
    md_lines.append(f"- Architecture Decisions: {counts.get('architecture', 0)}")
    md_lines.append(f"- Design Decisions: {counts.get('design', 0)}")
    md_lines.append(f"- Errors: {counts.get('errors', 0)}")
    md_lines.append(f"- Pivots: {counts.get('pivots', 0)}")
    md_lines.append(f"- Reasoning Events: {counts.get('reasoning', 0)}\n")
    md_lines.append("**Major Technical Themes**:")
    md_lines.append("- embeddings\n- keyword extraction\n- clustering\n- intent generation\n")
    md_lines.append("---\n")

    # Major Milestones
    if milestones:
        md_lines.append("## Major Milestones\n")
        for m in milestones:
            md_lines.append(f"- **{m['timestamp']}**: {m['text']}")
        md_lines.append("\n---\n")

    # Model Evolution (Versions)
    if versions:
        md_lines.append("## Model Evolution\n")
        for v in versions:
            md_lines.append(f"### {v['version']}")
            md_lines.append(f"{v['text']}\n")
        md_lines.append("---\n")

    # Key Engineering Learnings
    if lessons:
        md_lines.append("## Engineering Learnings\n")
        lesson_groups = defaultdict(list)
        for l in lessons:
            lesson_groups[l["type"]].append(l)
        for l_type, l_list in lesson_groups.items():
            md_lines.append(f"### {l_type}")
            for l in l_list:
                md_lines.append(f"- {l['text']}")
            md_lines.append("")
        md_lines.append("---\n")

    # Phase & Topic output
    
    for phase_name in ordered_phases:
        if phase_name in phases_dict and phases_dict[phase_name]:
            md_lines.append(f"# Phase: {phase_name}\n")
            
            for concept in phases_dict[phase_name]:
                md_lines.append(f"## Topic: {concept['title']}")
                md_lines.append(f"**Started**: {concept['primary_timestamp']} | **Topic ID**: `{concept['concept_id']}`\n")
                
                for r in concept["records"]:
                    cat = r["category"]
                    ts = r["timestamp"]
                    
                    if cat == "timeline":
                        event = r['raw'].get('event', r['text'])
                        md_lines.append(f"- **[TIMELINE]** ({ts}): {event}")
                    elif cat in ["architecture", "design"]:
                        decision = r["raw"].get("decision", r["text"])
                        reasoning = r["raw"].get("reasoning", "")
                        impact = r["raw"].get("impact", "")
                        md_lines.append(f"- **[{cat.upper()}]** {decision}")
                        if reasoning:
                            md_lines.append(f"  - *Why*: {reasoning}")
                        if impact:
                            md_lines.append(f"  - *Impact*: {impact}")
                    elif cat == "reasoning":
                        reason = r['raw'].get('reasoning', r['text'])
                        md_lines.append(f"  - *Reasoning*: {reason}")
                    elif cat == "pivots":
                        before = r["raw"].get("before", "")
                        after = r["raw"].get("after", "")
                        trigger = r["raw"].get("trigger", "")
                        if before and after:
                            md_lines.append(f"  - *Pivot*: {before} → {after}")
                        else:
                            md_lines.append(f"  - *Pivot*: {r['text']}")
                        if trigger:
                            md_lines.append(f"    - *Trigger*: {trigger}")
                    elif cat == "errors":
                        problem = r["raw"].get("problem", r["text"])
                        resolution = r["raw"].get("resolution", "")
                        lesson = r["raw"].get("lesson", "")
                        md_lines.append(f"  - *Debugging*: {problem}")
                        if resolution:
                            md_lines.append(f"    - *Resolution*: {resolution}")
                        if lesson:
                            md_lines.append(f"    - *Lesson*: {lesson}")
                
                md_lines.append("\n---\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Master document successfully generated at:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
