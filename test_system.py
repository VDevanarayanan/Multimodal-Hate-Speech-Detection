import unittest
import torch
import os
from config import CONFIG
from text_deobfuscator import TextDeobfuscator
from models.ddcaf_fusion import BidirectionalCrossAttention, DynamicContextGate
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from dataset_loader import MultimodalHateSpeechDataset
from evaluate import Evaluator

class TestMultimodalHateSpeechSystem(unittest.TestCase):
    
    def test_text_deobfuscator(self):
        self.assertEqual(TextDeobfuscator.clean("h@t3 speech"), "hate speech")
        self.assertEqual(TextDeobfuscator.clean("r.a.c.i.s.t"), "racist")
        self.assertEqual(TextDeobfuscator.clean("f\u200bu\u200bc\u200bk"), "fuck")
        self.assertEqual(TextDeobfuscator.clean("һаte ѕреесh"), "hate speech")

    def test_ddcaf_cross_attention(self):
        B, T, V, D = 2, 10, 196, 768
        text_seq = torch.randn(B, T, D)
        vision_seq = torch.randn(B, V, D)
        
        ddcaf = BidirectionalCrossAttention(embed_dim=D, num_heads=4)
        tgva, vgta = ddcaf(text_seq, vision_seq)
        
        self.assertEqual(tgva.shape, (B, T, D))
        self.assertEqual(vgta.shape, (B, V, D))

    def test_dynamic_context_gate(self):
        B, D = 2, 768
        text_cls = torch.randn(B, D)
        vis_cls = torch.randn(B, D)
        tgva_pool = torch.randn(B, D)
        vgta_pool = torch.randn(B, D)
        post_cls = torch.randn(B, D)
        
        gate = DynamicContextGate(embed_dim=D)
        fused, gates = gate(text_cls, vis_cls, tgva_pool, vgta_pool, post_cls)
        
        self.assertEqual(fused.shape, (B, D))
        self.assertEqual(gates.shape, (B, 4))
        # Gates should be in range [0, 1]
        self.assertTrue((gates >= 0).all() and (gates <= 1).all())

    def test_end_to_end_model_forward(self):
        model = MultimodalHateSpeechModel(CONFIG)
        text_ids = torch.randint(0, 1000, (2, 32))
        text_mask = torch.ones(2, 32)
        pixel_vals = torch.randn(2, 3, 224, 224)
        post_ids = torch.randint(0, 1000, (2, 32))
        
        outputs = model(text_ids, text_mask, pixel_vals, post_ids)
        self.assertEqual(outputs["binary_logits"].shape, (2, 2))
        self.assertEqual(outputs["multilabel_logits"].shape, (2, CONFIG.num_categories))

if __name__ == "__main__":
    unittest.main()
