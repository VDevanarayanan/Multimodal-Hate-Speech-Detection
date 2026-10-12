import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    # Model backbones
    text_model_name: str = "xlm-roberta-base"
    vision_model_name: str = "google/vit-base-patch16-224-in21k"
    
    # Hidden & Embedding dimensions
    text_embed_dim: int = 768
    vision_embed_dim: int = 768
    fusion_hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout_prob: float = 0.2
    
    # 8 Hate Speech Categories
    hate_categories: List[str] = field(default_factory=lambda: [
        "Racism / Xenophobia",
        "Sexism / Misogyny",
        "Homophobia / LGBTQ+ Hate",
        "Religious Hate",
        "Disability Discrimination",
        "Political Extremism",
        "Cyberbullying / Threat",
        "Dehumanization / Slurs"
    ])
    num_categories: int = 8
    
    # Rationale extraction max length
    max_text_len: int = 128
    
    # Training & Loss weights
    binary_loss_weight: float = 1.0
    multilabel_loss_weight: float = 1.0
    rationale_loss_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    import torch
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

CONFIG = ModelConfig()
