import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from config import CONFIG
from torch.utils.data import DataLoader
from train_facebook_fast import MemmapMultimodalDataset, forward_cached, MMAP_DIR

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ckpt = torch.load("checkpoints/best_hateful_memes.pt", map_location=device, weights_only=False)

model = MultimodalHateSpeechModel(CONFIG).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

dev_ds = MultimodalHateSpeechDataset(zip_path="Archive (1).zip", split_ids_file="data/dev.jsonl")
N = len(dev_ds)
dev_patches = np.memmap(f"{MMAP_DIR}/dev_patches.dat", dtype="float16", mode="r", shape=(N, 196, 768))
dev_cls = np.memmap(f"{MMAP_DIR}/dev_cls.dat", dtype="float16", mode="r", shape=(N, 768))
cached_dev = MemmapMultimodalDataset(dev_ds, dev_patches, dev_cls)
dev_loader = DataLoader(cached_dev, batch_size=32, shuffle=False)

probs_mm = []
probs_txt = []
probs_vis = []
gts = []

with torch.no_grad():
    for b in dev_loader:
        t_ids = b["text_input_ids"].to(device)
        t_mask = b["text_attention_mask"].to(device)
        patches = b["patch_tokens"].to(device)
        v_cls = b["vision_cls"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=(device.type == "mps")):
            o = forward_cached(model, t_ids, t_mask, patches, v_cls)
            pr_mm = torch.softmax(o["binary_logits"], dim=-1)[:, 1].cpu().numpy()
            pr_txt = torch.softmax(o["text_aux_logits"], dim=-1)[:, 1].cpu().numpy()
            pr_vis = torch.softmax(o["vision_aux_logits"], dim=-1)[:, 1].cpu().numpy()
        probs_mm.extend(pr_mm)
        probs_txt.extend(pr_txt)
        probs_vis.extend(pr_vis)
        gts.extend(b["binary_label"].numpy())

probs_mm = np.array(probs_mm)
probs_txt = np.array(probs_txt)
probs_vis = np.array(probs_vis)
gts = np.array(gts)

opt_t = ckpt.get("optimal_threshold", 0.36)
preds = (probs_mm >= opt_t).astype(int)

print(f"Optimal threshold from ckpt: {opt_t:.2f}")
print(f"Accuracy at {opt_t:.2f}: {accuracy_score(gts, preds)*100:.2f}%")
print(f"AUC: {roc_auc_score(gts, probs_mm):.4f}")
print("Confusion Matrix:")
cm = confusion_matrix(gts, preds)
print(cm)

# Check text-only and vision-only aux heads
acc_txt = accuracy_score(gts, (probs_txt >= 0.5).astype(int))
auc_txt = roc_auc_score(gts, probs_txt)
print(f"Text Aux Acc: {acc_txt*100:.2f}%, AUC: {auc_txt:.4f}")

# Test ensemble combinations
print("\nTesting Ensemble of Multimodal + Text Aux:")
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = (1 - w) * probs_mm + w * probs_txt
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        cur = accuracy_score(gts, (blend >= t).astype(int))
        if cur > best_acc:
            best_acc = cur
            best_t = t
    print(f"Blend Weight text={w:.1f} -> Best Acc: {best_acc*100:.2f}% (T={best_t:.2f}) | AUC: {roc_auc_score(gts, blend):.4f}")
