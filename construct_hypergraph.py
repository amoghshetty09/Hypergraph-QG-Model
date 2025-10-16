import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def construct_hypergraph(example):
    # Extract relevant fields
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]
    labels = example["labels"]
    answer = example["answer"]

    # 🔧 Simulate hypergraph nodes: treat each token span (chunk) as a node
    # You may later replace this with real entity, phrase, or sentence-level nodes
    num_tokens = len(input_ids)
    chunk_size = 8
    q_nodes = [input_ids[i:i+chunk_size] for i in range(0, num_tokens, chunk_size)]

    # Simulate knowledge nodes (same as question nodes here, can be extended)
    k_nodes = q_nodes.copy()

    # Construct hyperedges (dummy: connect node i with i+1)
    hyperedges = []
    for i in range(len(q_nodes) - 1):
        hyperedges.append([i, i + 1])  # connect consecutive chunks

    return {
        "q_nodes": q_nodes,
        "k_nodes": k_nodes,
        "hyperedges": hyperedges,
        "labels": labels,
        "answer": answer
    }

def main(args):
    data = load_data(args.input)
    processed = [construct_hypergraph(ex) for ex in tqdm(data)]

    save_data(args.output, processed)
    print(f"Saved {len(processed)} hypergraphs to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed hotpot file")
    parser.add_argument("--output", type=str, required=True, help="Path to save hypergraph file")
    args = parser.parse_args()

    main(args)
