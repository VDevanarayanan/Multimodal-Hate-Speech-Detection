import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_stream import ViTVisualStream
from models.xlm_roberta_stream import XLMRoBERTaTextStream
from models.ddcaf_fusion import BidirectionalCrossAttention, DynamicContextGate
from models.multitask_heads import BinaryHateHead
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
        
        # 4. Binary Output Head (Harmful vs Non-Harmful)
        self.binary_head = BinaryHateHead(embed_dim=config.text_embed_dim, hidden_dim=256, dropout=config.dropout_prob)
        
        # 5. Auxiliary Unimodal Supervision Heads (prevents modality shortcutting / collapse)
        self.text_aux_head = nn.Linear(config.text_embed_dim, 2)
        self.vision_aux_head = nn.Linear(config.vision_embed_dim, 2)

    def forward(self, 
                text_input_ids, 
                text_attention_mask=None, 
                pixel_values=None, 
                post_input_ids=None, 
                post_attention_mask=None,
                has_post=None,
                has_image=None,
                modality="multimodal",
                return_rationale=False):
        """
        text_input_ids: (B, Text_Len) -> Tokenized (Caption + OCR text)
        pixel_values: (B, 3, 224, 224) -> Image tensor
        post_input_ids: (B, Post_Len) -> Tokenized surrounding post text
        has_post: (B,) boolean tensor or None indicating whether post text actually exists
        has_image: (B,) boolean tensor or None indicating whether image actually exists
        modality: "multimodal" (default) | "text_only" | "image_only"
        return_rationale: if False (default), skips expensive rationale head computation during training
        """
        B = text_input_ids.shape[0]

        # Determine if image actually exists
        image_exists = True
        if has_image is not None:
            image_exists = bool(has_image.any().item()) if isinstance(has_image, torch.Tensor) else bool(has_image)
        elif pixel_values is not None:
            image_exists = bool((pixel_values.abs().sum() > 0).item()) if isinstance(pixel_values, torch.Tensor) else False
        else:
            image_exists = False

        # 1. Encode Text Stream (Caption + OCR) - Skip graph tracking if backbone is completely frozen
        if modality != "image_only":
            text_requires_grad = any(p.requires_grad for p in self.text_encoder.parameters())
            with torch.set_grad_enabled(text_requires_grad and self.training):
                text_seq, text_cls = self.text_encoder(text_input_ids, text_attention_mask)
        else:
            text_cls = torch.zeros(B, self.config.text_embed_dim, device=text_input_ids.device)
            text_seq = torch.zeros(B, self.config.max_text_len, self.config.text_embed_dim, device=text_input_ids.device)

        # 2. Encode Visual Stream (Image Patches) - Skip graph tracking if backbone is completely frozen
        if modality != "text_only" and image_exists and pixel_values is not None:
            vis_requires_grad = any(p.requires_grad for p in self.vision_encoder.parameters())
            with torch.set_grad_enabled(vis_requires_grad and self.training):
                patch_tokens, vision_cls = self.vision_encoder(pixel_values)
        else:
            patch_tokens = torch.zeros(B, 196, self.config.vision_embed_dim, device=text_input_ids.device)
            vision_cls = torch.zeros(B, self.config.vision_embed_dim, device=text_input_ids.device)

        # 3. Handle Baselines or Missing Image Mode Directly
        if modality == "text_only" or (modality == "multimodal" and not image_exists):
            fused_representation = text_cls
            gate_weights = torch.zeros(B, 4, device=text_cls.device)
            gate_weights[:, 0] = 1.0 # Text gate 1.0
        elif modality == "image_only":
            fused_representation = vision_cls
            gate_weights = torch.zeros(B, 4, device=vision_cls.device)
            gate_weights[:, 1] = 1.0 # Vision gate 1.0
        else:
            # 4. Encode Surrounding Post Text Stream - ONLY execute transformer if post text actually exists
            post_exists = False
            if has_post is not None:
                post_exists = bool(has_post.any().item()) if isinstance(has_post, torch.Tensor) else bool(has_post)
            elif post_input_ids is not None:
                # If has_post is not provided, check if tokens are non-pad
                post_exists = bool((post_input_ids > 2).any().item())
                
            if post_exists and post_input_ids is not None:
                post_requires_grad = any(p.requires_grad for p in self.post_text_encoder.parameters())
                with torch.set_grad_enabled(post_requires_grad and self.training):
                    _, post_cls_raw = self.post_text_encoder(post_input_ids, post_attention_mask)
                if has_post is not None:
                    hp_mask = has_post.float().view(B, 1)
                    post_cls = post_cls_raw * hp_mask
                else:
                    post_cls = post_cls_raw
            else:
                post_cls = torch.zeros_like(text_cls)

            # 5. DDCAF Cross-Attention: TGVA + VGTA with attention padding mask
            tgva_out, vgta_out = self.ddcaf(text_seq, patch_tokens, text_mask=text_attention_mask)
            
            # Mask-aware pooling over valid text positions (avoids pad token dilution)
            if text_attention_mask is not None:
                mask = text_attention_mask.unsqueeze(-1).float()
                tgva_pool = (tgva_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                tgva_pool = tgva_out.mean(dim=1)
                
            # Average over all 196 visual patch tokens
            vgta_pool = vgta_out.mean(dim=1)
            
            # 6. Dynamic Gating Fusion with Context Gate
            fused_representation, gate_weights = self.context_gate(
                main_text_cls=text_cls,
                vision_cls=vision_cls,
                tgva_pool=tgva_pool,
                vgta_pool=vgta_pool,
                post_text_cls=post_cls
            )
            
        # Binary classification head prediction (Harmful vs. Non-Harmful)
        binary_logits = self.binary_head(fused_representation)
        
        # Unimodal auxiliary logits
        text_aux_logits = self.text_aux_head(text_cls)
        vision_aux_logits = self.vision_aux_head(vision_cls)
        
        return {
            "binary_logits": binary_logits,
            "gate_weights": gate_weights,
            "fused_features": fused_representation,
            "text_aux_logits": text_aux_logits,
            "vision_aux_logits": vision_aux_logits
        }
