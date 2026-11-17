import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from config import CONFIG
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ckpt = torch.load("checkpoints/best_harmeme.pt", map_location=device, weights_only=False)

model = MultimodalHateSpeechModel(CONFIG).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

test_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="test.jsonl")
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

probs = []
gts = []

with torch.no_grad():
    for b in test_loader:
        t_ids = b["text_input_ids"].to(device)
        t_mask = b["text_attention_mask"].to(device)
        p_vals = b["pixel_values"].to(device)
        has_img = b["has_image"].to(device)
        
        out = model(
            text_input_ids=t_ids,
            text_attention_mask=t_mask,
            pixel_values=p_vals,
            has_image=has_img,
            modality="multimodal"
        )
        pr = torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy()
        probs.extend(pr)
        gts.extend(b["binary_label"].numpy())

probs = np.array(probs)
gts = np.array(gts)

opt_t = ckpt.get("optimal_threshold", 0.71)
preds = (probs >= opt_t).astype(int)

print(f"Optimal threshold from ckpt: {opt_t:.2f}")
print(f"Accuracy at {opt_t:.2f}: {accuracy_score(gts, preds)*100:.2f}%")
print(f"AUC: {roc_auc_score(gts, probs):.4f}")
print("Confusion Matrix:")
cm = confusion_matrix(gts, preds)
print(cm)
print("Classification Report:")
print(classification_report(gts, preds, target_names=["clean", "harmful"]))

# Threshold search
print("\nThreshold sweep:")
for t in np.arange(0.40, 0.85, 0.02):
    p = (probs >= t).astype(int)
    acc = accuracy_score(gts, p)
    print(f"T={t:.2f} -> Acc: {acc*100:.2f}% | Clean: {((p==0)&(gts==0)).sum()}/{ (gts==0).sum() } | Harm: {((p==1)&(gts==1)).sum()}/{ (gts==1).sum() }")
