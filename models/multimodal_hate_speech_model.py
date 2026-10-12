import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_stream import ViTVisualStream
from models.xlm_roberta_stream import XLMRoBERTaTextStream
from models.ddcaf_fusion import BidirectionalCrossAttention, DynamicContextGate
from models.multitask_heads import BinaryHateHead, MultiLabelHateHead, RationaleHead
from text_deobfuscator import TextDeobfuscator

class MultimodalHateSpeechModel(nn.Module):
    """
    Unified Multimodal Multilingual Hate Speech Detection System featuring DDCAF & Dynamic Gating.
    Implements exact architecture from problem specification:
    - Text Deobfuscator + Multilingual XLM-RoBERTa (Caption + OCR text)
    - ViT Visual Stream (Image patch tokens)
    - Post XLM-RoBERTa Stream (Surrounding post text)
    - Bidirectional Cross-Attention (TGVA + VGTA)
    - Dynamic Context Gate (Adaptive noise suppression for post text)
    - Multi-task output heads: Binary, 8-Category Multi-label, Rationale extraction
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Text & Vision Encoders
        self.text_encoder = XLMRoBERTaTextStream(model_name=config.text_model_name, embed_dim=config.text_embed_dim)
        self.vision_encoder = ViTVisualStream(model_name=config.vision_model_name, embed_dim=config.vision_embed_dim)
        self.post_text_encoder = XLMRoBERTaTextStream(model_name=config.text_model_name, embed_dim=config.text_embed_dim)
        
        # 2. DDCAF Fusion Core (TGVA + VGTA)
        self.ddcaf = BidirectionalCrossAttention(
            embed_dim=config.text_embed_dim, 
            num_heads=config.num_attention_heads, 
            dropout=config.dropout_prob
        )
        
        # 3. Dynamic Context Gating
        self.context_gate = DynamicContextGate(
            embed_dim=config.text_embed_dim, 
            hidden_dim=config.fusion_hidden_dim, 
            dropout=config.dropout_prob
        )
        
        # 4. Multi-Task Heads
        self.binary_head = BinaryHateHead(embed_dim=config.text_embed_dim, hidden_dim=256, dropout=config.dropout_prob)
        self.multilabel_head = MultiLabelHateHead(embed_dim=config.text_embed_dim, num_categories=config.num_categories, hidden_dim=256, dropout=config.dropout_prob)
        self.rationale_head = RationaleHead(embed_dim=config.text_embed_dim, hidden_dim=256, dropout=config.dropout_prob)

    def forward(self, 
                text_input_ids, 
                text_attention_mask=None, 
                pixel_values=None, 
                post_input_ids=None, 
                post_attention_mask=None):
        """
        text_input_ids: (B, Text_Len) -> Tokenized (Caption + OCR text)
        pixel_values: (B, 3, 224, 224) -> Image tensor
        post_input_ids: (B, Post_Len) -> Tokenized surrounding post text
        """
        # Encode Text Stream (Caption + OCR)
        text_seq, text_cls = self.text_encoder(text_input_ids, text_attention_mask)
        
        # Encode Visual Stream (Image Patches)
        if pixel_values is not None:
            patch_tokens, vision_cls = self.vision_encoder(pixel_values)
        else:
            B = text_input_ids.shape[0]
            patch_tokens = torch.zeros(B, 196, self.config.vision_embed_dim, device=text_input_ids.device)
            vision_cls = torch.zeros(B, self.config.vision_embed_dim, device=text_input_ids.device)
            
        # Encode Surrounding Post Text Stream
        if post_input_ids is not None:
            _, post_cls = self.post_text_encoder(post_input_ids, post_attention_mask)
        else:
            post_cls = torch.zeros_like(text_cls)

        # DDCAF Cross-Attention: TGVA + VGTA
        tgva_out, vgta_out = self.ddcaf(text_seq, patch_tokens, text_mask=text_attention_mask)
        
        # Pooling TGVA & VGTA representations
        tgva_pool = tgva_out.mean(dim=1)
        vgta_pool = vgta_out.mean(dim=1)
        
        # Dynamic Gating Fusion with Context Gate
        fused_representation, gate_weights = self.context_gate(
            main_text_cls=text_cls,
            vision_cls=vision_cls,
            tgva_pool=tgva_pool,
            vgta_pool=vgta_pool,
            post_text_cls=post_cls
        )
        
        # Multi-task head predictions
        binary_logits = self.binary_head(fused_representation)
        multilabel_logits = self.multilabel_head(fused_representation)
        rationale_logits = self.rationale_head(fused_representation, text_seq)
        
        return {
            "binary_logits": binary_logits,
            "multilabel_logits": multilabel_logits,
            "rationale_logits": rationale_logits,
            "gate_weights": gate_weights,
            "fused_features": fused_representation
        }
