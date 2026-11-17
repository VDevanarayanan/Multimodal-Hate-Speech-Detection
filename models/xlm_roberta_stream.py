import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class XLMRoBERTaTextStream(nn.Module):
    """
    Teal & Purple Text Stream: Multilingual XLM-RoBERTa encoder.
    Processes:
    1. RoBERTa text (caption + OCR text)
    2. Post RoBERTa (surrounding post text)
    Outputs sequence embeddings (B, Seq_Len, Text_Embed_Dim) and pooled CLS embedding (B, Text_Embed_Dim).
    """
    def __init__(self, model_name="xlm-roberta-base", embed_dim=768):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        try:
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        except Exception:
            try:
                self.encoder = AutoModel.from_pretrained(model_name)
            except Exception:
                # Fallback embedding layer if offline
                self.encoder = None
                self.word_embeddings = nn.Embedding(250002, embed_dim)
                self.pos_embeddings = nn.Embedding(512, embed_dim)
                self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, Seq_Len)
        attention_mask: (B, Seq_Len)
        Returns:
            sequence_embeddings: (B, Seq_Len, D_text)
            pooled_embedding: (B, D_text) - combines CLS token and attention-masked mean pooling
        """
        if self.encoder is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq_embeds = outputs.last_hidden_state # (B, Seq_Len, 768)
            cls_embed = seq_embeds[:, 0, :]        # (B, 768)
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                mean_embed = (seq_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                # Keyword-sensitive max pooling over valid sequence tokens
                masked_seq = seq_embeds.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
                max_embed = masked_seq.max(dim=1)[0]
                # Multi-pooling: CLS + semantic mean + keyword max pooling
                pooled_embed = (cls_embed + mean_embed + max_embed) / 3.0
            else:
                pooled_embed = cls_embed
                
            return seq_embeds, pooled_embed
        else:
            B, S = input_ids.shape
            pos = torch.arange(S, device=input_ids.device).unsqueeze(0).repeat(B, 1)
            x = self.word_embeddings(input_ids) + self.pos_embeddings(pos)
            x = self.layer_norm(x)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                mean_pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                masked_x = x.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
                max_pooled = masked_x.max(dim=1)[0]
                pooled = (x[:, 0, :] + mean_pooled + max_pooled) / 3.0
            else:
                pooled = x[:, 0, :]
            return x, pooled
