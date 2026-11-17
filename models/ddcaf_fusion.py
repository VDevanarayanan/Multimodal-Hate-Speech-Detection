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
        
        # Feature scale normalization: projects disparate text and vision norm spaces to unit variance
        self.text_in_norm = nn.LayerNorm(embed_dim)
        self.vis_in_norm = nn.LayerNorm(embed_dim)
        
        # Text-Guided Visual Attention (TGVA)
        self.tgva_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.tgva_norm = nn.LayerNorm(embed_dim)
        
        # Visual-Guided Text Attention (VGTA)
        self.vgta_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.vgta_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    @staticmethod
    def _fast_attention(mha_module, q, k, v, key_valid_mask=None, dropout_p=0.0, training=False):
        B, L_q, D = q.shape
        L_k = k.shape[1]
        H = mha_module.num_heads
        d_k = D // H
        
        if mha_module._qkv_same_embed_dim:
            w_q, w_k, w_v = mha_module.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = (mha_module.in_proj_bias.chunk(3, dim=0) if mha_module.in_proj_bias is not None else (None, None, None))
            q_proj = F.linear(q, w_q, b_q).view(B, L_q, H, d_k).transpose(1, 2)
            k_proj = F.linear(k, w_k, b_k).view(B, L_k, H, d_k).transpose(1, 2)
            v_proj = F.linear(v, w_v, b_v).view(B, L_k, H, d_k).transpose(1, 2)
        else:
            q_proj = mha_module.q_proj(q).view(B, L_q, H, d_k).transpose(1, 2)
            k_proj = mha_module.k_proj(k).view(B, L_k, H, d_k).transpose(1, 2)
            v_proj = mha_module.v_proj(v).view(B, L_k, H, d_k).transpose(1, 2)

        attn_mask = None
        if key_valid_mask is not None:
            mask_clone = key_valid_mask.bool().clone()
            all_false = (~mask_clone).all(dim=-1)
            if all_false.any():
                mask_clone[all_false, 0] = True
            attn_mask = mask_clone.view(B, 1, 1, L_k).expand(B, H, L_q, L_k)

        drop = dropout_p if training else 0.0
        out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=attn_mask, dropout_p=drop)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        return mha_module.out_proj(out)

    def forward(self, text_seq, vision_seq, text_mask=None):
        """
        text_seq: (B, Text_Len, D)
        vision_seq: (B, Patch_Len, D)
        text_mask: (B, Text_Len) binary mask (1 for valid token, 0 for pad)
        Returns:
            tgva_out: (B, Text_Len, D) -> Text attend to Visual patches
            vgta_out: (B, Patch_Len, D) -> Visual patches attend to Text
        """
        # Normalize representations so attention dot products are well-scaled
        t_seq = self.text_in_norm(text_seq)
        v_seq = self.vis_in_norm(vision_seq)
        
        # TGVA: Q = Text, K, V = Vision (Visual patches are all valid)
        tgva_feat = self._fast_attention(self.tgva_attn, t_seq, v_seq, v_seq, dropout_p=self.dropout, training=self.training)
        tgva_out = self.tgva_norm(text_seq + tgva_feat)
        
        # VGTA: Q = Vision, K, V = Text (Mask out pad tokens in text)
        if text_mask is not None:
            tgva_out = tgva_out * text_mask.unsqueeze(-1).float()
            
        vgta_feat = self._fast_attention(self.vgta_attn, v_seq, t_seq, t_seq, key_valid_mask=text_mask, dropout_p=self.dropout, training=self.training)
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
        
        # Modality scale normalization: balances text and vision CLS feature magnitudes
        self.text_norm = nn.LayerNorm(embed_dim)
        self.vis_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        
        # Gating network
        # Inputs: [text_cls, vision_cls, tgva_pool, vgta_pool, post_text_cls, hadamard] (6 * embed_dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(embed_dim * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # 4 gating weights: [main_text, vision, cross_attn, post_text]
            nn.Sigmoid()
        )
        
        # Feature projection layer with Bilinear Cross-Modal Interaction
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.out_norm = nn.LayerNorm(embed_dim)

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
            
        # Normalize representations to unit variance so neither modality dominates
        norm_text = self.text_norm(main_text_cls)
        norm_vis = self.vis_norm(vision_cls)
        norm_post = self.post_norm(post_text_cls)
        hadamard = norm_text * norm_vis  # Bilinear cross-modal interaction
            
        concat_gate_inputs = torch.cat([norm_text, norm_vis, tgva_pool, vgta_pool, norm_post, hadamard], dim=-1)
        gates = self.gate_fc(concat_gate_inputs) # (B, 4)
        
        # Anti-collapse bounded gating: Ensure text, vision, and cross-attention maintain minimum floor (>=0.20)
        # Prevents unimodal dominance where one branch starves the other of gradients
        g_text = 0.25 + 0.75 * gates[:, 0:1]
        g_vis = 0.25 + 0.75 * gates[:, 1:2]
        g_cross = 0.20 + 0.80 * gates[:, 2:3]
        g_post = gates[:, 3:4] # Suppresses noisy post text if low
        
        cross_attn_combined = (tgva_pool + vgta_pool) / 2.0
        
        weighted_text = g_text * norm_text
        weighted_vis = g_vis * norm_vis
        weighted_cross = g_cross * cross_attn_combined
        weighted_post = g_post * norm_post
        weighted_hadamard = g_cross * hadamard
        
        fused_raw = torch.cat([weighted_text, weighted_vis, weighted_cross, weighted_post, weighted_hadamard], dim=-1)
        fused_vec = self.fusion_projection(fused_raw)
        
        # Multimodal residual highway: preserves balanced text, vision, and cross-attention signals directly
        fused_vec = self.out_norm(fused_vec + 0.5 * norm_text + 0.5 * norm_vis + 0.25 * cross_attn_combined)
        
        effective_gates = torch.cat([g_text, g_vis, g_cross, g_post], dim=-1)
        return fused_vec, effective_gates
