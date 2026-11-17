import os
import sys

# Auto-fallback to local virtual environment python if torch is missing in current environment
_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
if os.path.exists(_venv_py) and os.path.realpath(sys.executable) != os.path.realpath(_venv_py):
    try:
        import torch
    except ImportError:
        os.execv(_venv_py, [_venv_py] + sys.argv)

import unittest
import torch
import numpy as np
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
        # Ensure numbers, dollar amounts, and valid English letters ('v') are preserved
        self.assertEqual(TextDeobfuscator.clean("order $37.50 white privilege"), "order $37.50 white privilege")
        self.assertEqual(TextDeobfuscator.clean("year 2020 100% video"), "year 2020 100% video")

    def test_ddcaf_cross_attention(self):
        B, T, V, D = 2, 10, 196, 768
        text_seq = torch.randn(B, T, D)
        vision_seq = torch.randn(B, V, D)
        text_mask = torch.ones(B, T)
        
        ddcaf = BidirectionalCrossAttention(embed_dim=D, num_heads=4)
        tgva, vgta = ddcaf(text_seq, vision_seq, text_mask=text_mask)
        
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
        self.assertIn("gate_weights", outputs)

    def test_baseline_modalities(self):
        model = MultimodalHateSpeechModel(CONFIG)
        text_ids = torch.randint(0, 1000, (2, 32))
        text_mask = torch.ones(2, 32)
        pixel_vals = torch.randn(2, 3, 224, 224)
        
        out_text = model(text_ids, text_mask, pixel_vals, modality="text_only")
        out_img = model(text_ids, text_mask, pixel_vals, modality="image_only")
        
        self.assertEqual(out_text["binary_logits"].shape, (2, 2))
        self.assertEqual(out_img["binary_logits"].shape, (2, 2))
        # Verify gate weights reflect text-only / image-only
        self.assertEqual(out_text["gate_weights"][0, 0].item(), 1.0)
        self.assertEqual(out_img["gate_weights"][0, 1].item(), 1.0)

    def test_optimal_threshold_finder(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        thresh, score = Evaluator.find_optimal_threshold(y_true, y_probs)
        self.assertTrue(0.4 <= thresh <= 0.6)
        self.assertAlmostEqual(score, 1.0, places=2)

if __name__ == "__main__":
    unittest.main()
