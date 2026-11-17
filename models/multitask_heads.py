import torch
import torch.nn as nn

class BinaryHateHead(nn.Module):
    """
    Binary Head: Classifies input into Hate Speech (1) vs Normal/Non-hate (0).
    Uses a streamlined 2-layer MLP with LayerNorm, GELU, and Dropout (0.2) to provide
    clean gradient flow directly to the backbones and fusion representations,
    preventing parameter memorization and maximizing test generalization.
    """
    def __init__(self, embed_dim=768, hidden_dim=256, dropout=0.2):
        super().__init__()
        actual_hidden = max(hidden_dim, 256)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, actual_hidden),
            nn.LayerNorm(actual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(actual_hidden, 2)
        )

    def forward(self, fused_features):
        return self.classifier(fused_features)


class MultiLabelHateHead(nn.Module):
    """
    Multi-label Head: Predicts multi-label probabilities across 8 hate categories:
    1. Racism / Xenophobia
    2. Sexism / Misogyny
    3. Homophobia / LGBTQ+ Hate
    4. Religious Hate
    5. Disability Discrimination
    6. Political Extremism
    7. Cyberbullying / Threat
    8. Dehumanization / Slurs
    """
    def __init__(self, embed_dim=768, num_categories=8, hidden_dim=512, dropout=0.2):
        super().__init__()
        actual_hidden = max(hidden_dim, 256)
        self.fc1 = nn.Linear(embed_dim, actual_hidden)
        self.ln1 = nn.LayerNorm(actual_hidden)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(actual_hidden, actual_hidden // 2)
        self.ln2 = nn.LayerNorm(actual_hidden // 2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout / 2.0)
        
        self.out = nn.Linear(actual_hidden // 2, num_categories)

    def forward(self, fused_features):
        h = self.drop1(self.act1(self.ln1(self.fc1(fused_features))))
        h = self.drop2(self.act2(self.ln2(self.fc2(h))))
        return self.out(h)


class RationaleHead(nn.Module):
    """
    Rationale Head: Verb-object phrase pointer / token score extraction head.
    Predicts token-level rationale logits over the input text tokens to extract explanations.
    """
    def __init__(self, embed_dim=768, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.project_fused = nn.Linear(embed_dim, embed_dim)
        self.rationale_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fused_features, text_sequence_embeddings):
        """
        fused_features: (B, D)
        text_sequence_embeddings: (B, Seq_Len, D)
        Returns:
            rationale_logits: (B, Seq_Len)
        """
        B, S, D = text_sequence_embeddings.shape
        fused_expanded = fused_features.unsqueeze(1).repeat(1, S, 1) # (B, S, D)
        combined = torch.cat([text_sequence_embeddings, fused_expanded], dim=-1) # (B, S, 2D)
        logits = self.rationale_classifier(combined).squeeze(-1) # (B, S)
        return logits
