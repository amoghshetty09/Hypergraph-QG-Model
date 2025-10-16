import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from tqdm import tqdm


class HotpotQGDataset(Dataset):
    def __init__(self, json_path, tokenizer=None, max_input_len=512, max_output_len=64, n_contexts=2):
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.n_contexts = n_contexts
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained("t5-small")

        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []

        print("Processing dataset...")
        for item in tqdm(data):
            question = item["question"]
            answer = item["answer"]
            supporting_titles = [sf[0] for sf in item["supporting_facts"]]

            # Select top-N supporting paragraphs (from context)
            context_chunks = []
            for title, paras in item["context"]:
                if title in supporting_titles and paras:
                    context_chunks.append(" ".join(paras))
                if len(context_chunks) >= n_contexts:
                    break

            if not context_chunks:
                continue

            # Concatenate multiple contexts
            combined_context = " ".join(context_chunks)

            # Input = context + answer
            decoder_input = f"answer: {answer}"
            target_question = f"{question}"

            # Tokenize
            context_ids = self.tokenizer.encode(
                combined_context,
                padding="max_length",
                truncation=True,
                max_length=max_input_len,
                return_tensors="pt"
            ).squeeze(0)

            answer_ids = self.tokenizer.encode(
                decoder_input,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            ).squeeze(0)

            question_ids = self.tokenizer.encode(
                target_question,
                padding="max_length",
                truncation=True,
                max_length=max_output_len,
                return_tensors="pt"
            ).squeeze(0)

            # Fake hypergraph input (just to test the model, later we generate real H_q and H_k)
            # Format: (H_q_nodes, H_k_nodes), each shape: [num_nodes]
            h_q = torch.randint(0, 1000, (8,))  # Dummy
            h_k = torch.randint(0, 1000, (10,))  # Dummy

            self.samples.append({
                "context_ids": context_ids,
                "answer_ids": answer_ids,
                "question_ids": question_ids,
                "h_q": h_q,
                "h_k": h_k,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample["h_q"].unsqueeze(0),     # he_ques
            sample["h_k"].unsqueeze(0),     # he_kg
            sample["answer_ids"],
            sample["question_ids"],
        )
