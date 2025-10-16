import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer
from models.hypergraph_qg_model import HypergraphQGModel
from datasets.hypergraph_qg_dataset import HypergraphQGDataset
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# === Config ===
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 3e-5
SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/hotpot_hypergraphs_5.json"

# === Custom collate function ===
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

# === Load dataset ===
dataset = HypergraphQGDataset(DATA_PATH)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate)

# === Model ===
model = HypergraphQGModel()
model.to(DEVICE)

# === Tokenizer (for decoding)
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# === Optimizer ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# === Training ===
os.makedirs(SAVE_DIR, exist_ok=True)

best_val_loss = float("inf")
patience = 30
trigger_times = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        q_nodes = batch["q_nodes"].to(DEVICE)
        k_nodes = batch["k_nodes"].to(DEVICE)
        hyperedges = batch["hyperedges"]
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        output = model(q_nodes, k_nodes, hyperedges, labels)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    # === Evaluation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            q_nodes = batch["q_nodes"].to(DEVICE)
            k_nodes = batch["k_nodes"].to(DEVICE)
            hyperedges = batch["hyperedges"]
            labels = batch["labels"].to(DEVICE)

            output = model(q_nodes, k_nodes, hyperedges, labels)
            val_loss += output["loss"].item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"📉 Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
        print("💾 Saved new best model.")
    else:
        trigger_times += 1
        print(f"⚠ No improvement. Early stop trigger count: {trigger_times}/{patience}")
        if trigger_times >= patience:
            print("🛑 Early stopping triggered.")
            break

    # === Generation sample ===
    sample = dataset[0]
    q_nodes = sample["q_nodes"].unsqueeze(0).to(DEVICE)
    k_nodes = sample["k_nodes"].unsqueeze(0).to(DEVICE)
    hyperedges = sample["hyperedges"]
    clamped_hyperedges = [
        [min(edge[0], q_nodes.size(1) - 1)] + [min(k, k_nodes.size(1) - 1) for k in edge[1:]]
        for edge in hyperedges
    ]
    output_ids = model.generate(q_nodes, k_nodes, [clamped_hyperedges], max_length=64)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("🧠 Sample Generation:", decoded)

# === Save Final Model ===
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
print("🎉 Training complete. Final model saved.")