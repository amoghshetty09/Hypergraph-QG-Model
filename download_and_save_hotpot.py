from datasets import load_dataset
import json
import os

# Load dataset
dataset = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)

# Extract train set
train_data = dataset["train"]

# Create minimal format compatible with our hypergraph preprocessor
converted_data = []
for item in train_data:
    converted_data.append({
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "type": item["type"],
        "level": item["level"],
        "supporting_facts": item["supporting_facts"],
        "context": item["context"]
    })

# Save to JSON
os.makedirs("data", exist_ok=True)
with open("data/hotpot_train.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("✅ Saved data/hotpot_train.json")
