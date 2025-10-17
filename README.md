# 🧠 Hypergraph-Enhanced T5 for Multi-Hop Question Generation

This repository presents an advanced **Hypergraph-Enhanced T5 model** for **multi-hop question generation (QG)** from unstructured text.  
The model integrates a **custom Hypergraph Transformer Encoder** with a **T5 decoder**, enabling reasoning across multiple context sentences through dynamic hyperedge connections.

---

## 🚀 Overview

Traditional Question Generation models struggle with **multi-hop reasoning**, where a question depends on connecting multiple facts or sentences.  
Our model introduces a **Hypergraph Encoder** that represents facts as nodes and dynamically connects them with **hyperedges** based on semantic similarity.  

The encoded representations are then decoded by a **T5 model** to produce coherent, context-aware questions.

---

## 🧩 Architecture

**Main Components**
1. 🧊 **Frozen T5 Encoder** – Extracts 768-dim embeddings from contextual sentences.  
2. 🕸 **Hypergraph Construction** – Builds top-K hyperedges between support and distractor facts using cosine similarity.  
3. 🔺 **Custom Hypergraph Encoder** – A stack of 3 Hypergraph Transformer layers performing attention over hyperedges.  
4. 🔁 **Linear Projection** – Maps 256-dim encoder output to 768-dim for T5 decoder compatibility.  
5. 🧠 **T5 Decoder** – Generates final questions using beam search decoding.

📊 *Key Difference:*  
Unlike vanilla T5, which treats text sequentially, this model performs **graph-based reasoning** over sentence relationships to improve multi-hop understanding.

---

## 📁 Project Structure



├── models/
│ └── hypergraph_qg_model.py # Combines Hypergraph Encoder and T5
│
├── modules/
│ ├── hypergraph_encoder.py # Stacked Hypergraph Transformer layers
│ └── hypergraph_transformer_layer.py # Implements hyperedge attention
│
├── train.py # Model training script
├── evaluate.py # Evaluation and question generation
├── requirements.txt # Dependencies list
├── Dockerfile # Docker setup file
└── README.md # You are here 🙂



---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Hypergraph-T5-QG.git
cd Hypergraph-T5-QG


2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ (Optional) Build Docker image
docker build -t hypergraph-qg .

🧪 Usage
▶️ Train the model
python train.py

🧠 Evaluate / Generate Questions
python evaluate.py

🐳 Run via Docker
docker run -it hypergraph-qg

