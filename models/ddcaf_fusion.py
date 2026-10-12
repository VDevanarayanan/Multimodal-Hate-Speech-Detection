import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalCrossAttention(nn.Module):
    """
    DDCAF Core: Dual-Directional Cross-Attention Fusion.
    Computes:
    1. TGVA (Text-Guided Visual Attention): Text queries visual patches.
    2. VGTA (Visual-Guided Text Attention): Visual patches query text tokens.
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Text-Guided Visual Attention (TGVA)
        self.tgva_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.tgva_norm = nn.LayerNorm(embed_dim)
        
        # Visual-Guided Text Attention (VGTA)
        self.vgta_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.vgta_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_seq, vision_seq, text_mask=None):
        """
        text_seq: (B, Text_Len, D)
        vision_seq: (B, Patch_Len, D)
        Returns:
            tgva_out: (B, Text_Len, D) -> Text attend to Visual patches
            vgta_out: (B, Patch_Len, D) -> Visual patches attend to Text
        """
        # TGVA: Q = Text, K, V = Vision
        tgva_feat, _ = self.tgva_attn(query=text_seq, key=vision_seq, value=vision_seq)
        tgva_out = self.tgva_norm(text_seq + tgva_feat)
        
        # VGTA: Q = Vision, K, V = Text
        key_padding_mask = (text_mask == 0) if text_mask is not None else None
        vgta_feat, _ = self.vgta_attn(query=vision_seq, key=text_seq, value=text_seq, key_padding_mask=key_padding_mask)
        vgta_out = self.vgta_norm(vision_seq + vgta_feat)
        
        return tgva_out, vgta_out


class DynamicContextGate(nn.Module):
    """
    Purple Stream: Dynamic Gating Fusion with Context Gate.
    Dynamically balances visual features, text features, cross-attention representations,
    and dynamically suppresses noisy surrounding post text.
    """
    def __init__(self, embed_dim=768, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Gating network
        # Inputs: [text_cls, vision_cls, tgva_pool, vgta_pool, post_text_cls] (5 * embed_dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(embed_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # 4 gating weights: [main_text, vision, cross_attn, post_text]
            nn.Sigmoid()
        )
        
        # Feature projection layer
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, main_text_cls, vision_cls, tgva_pool, vgta_pool, post_text_cls=None):
        """
        main_text_cls: (B, D)
        vision_cls: (B, D)
        tgva_pool: (B, D)
        vgta_pool: (B, D)
        post_text_cls: (B, D) optional
        Returns:
            fused_representation: (B, D)
            gate_weights: (B, 4) [main_text_gate, vision_gate, cross_attn_gate, post_text_gate]
        """
        B = main_text_cls.shape[0]
        if post_text_cls is None:
            post_text_cls = torch.zeros_like(main_text_cls)
            
        concat_gate_inputs = torch.cat([main_text_cls, vision_cls, tgva_pool, vgta_pool, post_text_cls], dim=-1)
        gates = self.gate_fc(concat_gate_inputs) # (B, 4)
        
        g_text = gates[:, 0:1]
        g_vis = gates[:, 1:2]
        g_cross = gates[:, 2:3]
        g_post = gates[:, 3:4] # Suppresses noisy post text if low
        
        cross_attn_combined = (tgva_pool + vgta_pool) / 2.0
        
        weighted_text = g_text * main_text_cls
        weighted_vis = g_vis * vision_cls
        weighted_cross = g_cross * cross_attn_combined
        weighted_post = g_post * post_text_cls
        
        fused_raw = torch.cat([weighted_text, weighted_vis, weighted_cross, weighted_post], dim=-1)
        fused_vec = self.fusion_projection(fused_raw)
        
        return fused_vec, gates
