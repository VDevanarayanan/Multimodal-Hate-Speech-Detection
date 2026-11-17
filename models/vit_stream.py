import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, ViTModel

class ViTVisualStream(nn.Module):
    """
    Coral Stream: ViT visual feature extractor.
    Extracts patch token representations (B, Num_Patches, Vision_Embed_Dim) 
    and CLS visual token (B, Vision_Embed_Dim).
    """
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", embed_dim=768):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        try:
            self.vit = ViTModel.from_pretrained(model_name, attn_implementation="eager")
        except Exception:
            try:
                self.vit = ViTModel.from_pretrained(model_name)
            except Exception:
                # Fallback transformer if offline or downloading is unavailable
                self.vit = None
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.randn(1, 197, embed_dim))

    def forward(self, pixel_values):
        """
        pixel_values: (B, 3, 224, 224) PyTorch Tensor
        Returns:
            patch_embeddings: (B, N_patches, D_vision)
            cls_embedding: (B, D_vision)
        """
        if self.vit is not None:
            outputs = self.vit(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state # (B, 197, 768)
            cls_token = last_hidden_state[:, 0, :]        # (B, 768)
            patch_tokens = last_hidden_state[:, 1:, :]    # (B, 196, 768)
            return patch_tokens, cls_token
        else:
            # Fallback projection
            B = pixel_values.shape[0]
            x = self.patch_embed(pixel_values)            # (B, D, 14, 14)
            x = x.flatten(2).transpose(1, 2)              # (B, 196, D)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)         # (B, 197, D)
            x = x + self.pos_embed[:, :x.size(1), :]
            return x[:, 1:, :], x[:, 0, :]
