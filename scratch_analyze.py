import json
from collections import Counter

def analyze_json(filename, out_f):
    with open(f"extracted/{filename}", 'r', encoding='utf-8') as f:
        data = json.load(f)
    out_f.write(f"\n=== {filename} (Total: {len(data)}) ===\n")
    
    # Filter out empty/None pivots
    if filename == "pivots.json":
        valid = [i for i in data if i.get('pivot') and str(i.get('pivot')).lower() != 'none']
        for item in valid[:15]:
            out_f.write(f"- Pivot: {item.get('pivot')} | Reason: {item.get('reason')}\n")
    elif filename == "architecture.json":
        for item in data[:15]:
            out_f.write(f"- Architecture: {item.get('decision')} | Rationale: {item.get('rationale')}\n")
    elif filename == "design.json":
        for item in data[:15]:
            out_f.write(f"- Design: {item.get('design_choice')} | Justification: {item.get('justification')}\n")
    elif filename == "timeline.json":
        for item in data[:15]:
            out_f.write(f"- Event: {item.get('event')} | Reason: {item.get('reason')}\n")

with open("summary.txt", "w", encoding="utf-8") as out_f:
    for file in ["pivots.json", "architecture.json", "design.json", "timeline.json"]:
        analyze_json(file, out_f)
