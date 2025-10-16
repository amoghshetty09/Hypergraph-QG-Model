import torch
from transformers import T5Tokenizer
from models.hypergraph_qg_model import HypergraphQGModel
from datasets.hypergraph_qg_dataset import HypergraphQGDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final.pt"
DATA_PATH = "data/hotpot_hypergraphs_new5k.json"  # or your actual dataset
BATCH_SIZE = 64  # Generate one-by-one for clarity

# === Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# === Load Dataset
dataset = HypergraphQGDataset(DATA_PATH)

def custom_collate(batch):
    q_nodes = [item["q_nodes"] for item in batch]
    k_nodes = [item["k_nodes"] for item in batch]
    hyperedges = [item["hyperedges"] for item in batch]
    labels = [item["labels"] for item in batch]

    q_nodes_padded = pad_sequence(q_nodes, batch_first=True)
    k_nodes_padded = pad_sequence(k_nodes, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "q_nodes": q_nodes_padded,
        "k_nodes": k_nodes_padded,
        "hyperedges": hyperedges,
        "labels": labels_padded
    }

loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)

# === Load Model
model = HypergraphQGModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Run Generation
print("🧪 Generating questions from saved model...\n")
with torch.no_grad():
    for i, batch in enumerate(loader):
        q_nodes = batch["q_nodes"].to(DEVICE)
        k_nodes = batch["k_nodes"].to(DEVICE)
        hyperedges = batch["hyperedges"]

        # Clamp hyperedges (in case of padding)
        clamped_hyperedges = []
        for b in range(len(hyperedges)):
            q_len = q_nodes.shape[1]
            k_len = k_nodes.shape[1]
            clamped_sample = []
            for edge in hyperedges[b]:
                q_idx = min(edge[0], q_len - 1)
                k_idxs = [min(k, k_len - 1) for k in edge[1:]]
                clamped_sample.append([q_idx] + k_idxs)
            clamped_hyperedges.append(clamped_sample)

        # Generate
        generated_ids = model.generate(q_nodes, k_nodes, clamped_hyperedges, max_length=64)
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"🔹 Sample {i+1}:")
        print("📤 Generated Question:", output_text)

        if i == 4:  # Limit to 5 samples
            break
