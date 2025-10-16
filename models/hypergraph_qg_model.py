import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from modules.hypergraph_encoder import HypergraphEncoder  # Import your encoder

class HypergraphQGModel(nn.Module):
    def __init__(self, 
                 encoder_hidden_size=64, 
                 num_heads=4, 
                 num_layers=3,
                 dropout=0.3,
                 t5_model_name="t5-base"):
        """
        Complete Hypergraph-to-Question Generation Model
        Args:
            encoder_hidden_size: Must match HypergraphEncoder (64)
            num_heads: Must divide encoder_hidden_size (4)
            num_layers: Number of encoder layers (3)
            dropout: Consistent with encoder (0.3)
            t5_model_name: Pretrained T5 variant
        """
        super().__init__()

        # 1. Configuration Validation
        assert encoder_hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # 2. Hypergraph Encoder (matches your architecture)
        self.encoder = HypergraphEncoder(
            hidden_size=encoder_hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 3. T5 Decoder Components
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_config = self.t5.config
        self.encoder_to_t5 = nn.Linear(encoder_hidden_size, self.t5_config.d_model)
        
        # 4. Generation Setup
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.prefix = "generate question:"
        
        # 5. Regularization (matches encoder dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_nodes, k_nodes, hyperedges, labels=None):
        """
        Forward pass for training
        Args:
            q_nodes: (B, Q, 64) query nodes
            k_nodes: (B, K, 64) key nodes  
            hyperedges: List[B] of hyperedge lists
            labels: (B, L) target token IDs
        """
        # 1. Hypergraph Encoding
        encoder_output = self.dropout(self.encoder(q_nodes, k_nodes, hyperedges))  # (B, Q, 64)
        projected = self.encoder_to_t5(encoder_output)  # (B, Q, d_model)
        
        # 2. Prepare T5 Inputs
        attention_mask = torch.ones(projected.size()[:2], 
                                  dtype=torch.long,
                                  device=projected.device)
        
        # 3. T5 Forward Pass
        return self.t5(
            inputs_embeds=torch.zeros((projected.size(0), 1, self.t5_config.d_model), 
                                    device=projected.device),
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def generate(self, q_nodes, k_nodes, hyperedges, max_length=64):
        """
        Generation with batch support
        Args:
            q_nodes: (B, Q, 64)
            k_nodes: (B, K, 64)
            hyperedges: List[B] of hyperedges
            max_length: Max output tokens
        """
        # 1. Encode Hypergraph
        encoder_output = self.encoder(q_nodes, k_nodes, hyperedges)  # (B, Q, 64)
        projected = self.encoder_to_t5(encoder_output)  # (B, Q, d_model)
        
        # 2. Prepare Decoder Inputs
        B = q_nodes.size(0)
        decoder_inputs = self.tokenizer(
            [self.prefix]*B,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(projected.device)
        
        # 3. Generate Questions
        return self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            decoder_input_ids=decoder_inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )