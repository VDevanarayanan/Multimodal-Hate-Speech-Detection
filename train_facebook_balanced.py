import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from config import CONFIG
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from dataset_loader import MultimodalHateSpeechDataset
from train_facebook_fast import MemmapMultimodalDataset, forward_cached, MMAP_DIR

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"===========================================================")
    print(f" Balanced Focal Facebook Hateful Memes Training ({device})")
    print(f"===========================================================")

    train_ds = MultimodalHateSpeechDataset(zip_path="Archive (1).zip", split_ids_file="data/train.jsonl")
    dev_ds = MultimodalHateSpeechDataset(zip_path="Archive (1).zip", split_ids_file="data/dev.jsonl")

    N_train = len(train_ds)
    N_dev = len(dev_ds)

    train_patches = np.memmap(f"{MMAP_DIR}/train_patches.dat", dtype="float16", mode="r", shape=(N_train, 196, 768))
    train_cls = np.memmap(f"{MMAP_DIR}/train_cls.dat", dtype="float16", mode="r", shape=(N_train, 768))

    dev_patches = np.memmap(f"{MMAP_DIR}/dev_patches.dat", dtype="float16", mode="r", shape=(N_dev, 196, 768))
    dev_cls = np.memmap(f"{MMAP_DIR}/dev_cls.dat", dtype="float16", mode="r", shape=(N_dev, 768))

    cached_train = MemmapMultimodalDataset(train_ds, train_patches, train_cls)
    cached_dev = MemmapMultimodalDataset(dev_ds, dev_patches, dev_cls)

    # Balanced 50/50 sampler
    train_labels = [s["label"] for s in train_ds.samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=N_train, replacement=True)

    train_loader = DataLoader(cached_train, batch_size=16, sampler=sampler)
    dev_loader = DataLoader(cached_dev, batch_size=32, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    # Freeze base model, unfreeze top 6 layers of XLM-R + DDCAF + ContextGate + BinaryHead
    for p in model.parameters():
        p.requires_grad = False

    xlm_layers = model.text_encoder.encoder.encoder.layer
    backbone_params = []
    for layer in xlm_layers[-6:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    head_params = []
    for module in [model.ddcaf, model.context_gate, model.binary_head, model.text_aux_head, model.vision_aux_head]:
        for p in module.parameters():
            p.requires_grad = True
            head_params.append(p)

    optimizer = AdamW([
        {'params': head_params, 'lr': 3e-4, 'weight_decay': 0.02},
        {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.01}
    ])

    epochs = 8
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    focal_criterion = FocalLoss(gamma=1.5, label_smoothing=0.03)
    aux_criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    amp_enabled = (device.type == "mps")
    best_dev_acc = 0.0
    best_payload = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            t_ids = batch["text_input_ids"].to(device)
            t_mask = batch["text_attention_mask"].to(device)
            patches = batch["patch_tokens"].to(device)
            v_cls = batch["vision_cls"].to(device)
            targets = batch["binary_label"].to(device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                out = forward_cached(model, t_ids, t_mask, patches, v_cls)
                loss_bin = focal_criterion(out["binary_logits"], targets)
                loss_t_aux = aux_criterion(out["text_aux_logits"], targets)
                loss = loss_bin + 0.15 * loss_t_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate on dev
        model.eval()
        probs_mm, gts = [], []
        with torch.no_grad():
            for b in dev_loader:
                t_ids = b["text_input_ids"].to(device)
                t_mask = b["text_attention_mask"].to(device)
                patches = b["patch_tokens"].to(device)
                v_cls = b["vision_cls"].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                    o = forward_cached(model, t_ids, t_mask, patches, v_cls)
                    pr = torch.softmax(o["binary_logits"], dim=-1)[:, 1].cpu().numpy()
                probs_mm.extend(pr)
                gts.extend(b["binary_label"].numpy())

        probs_mm = np.array(probs_mm)
        gts = np.array(gts)

        acc_05 = accuracy_score(gts, (probs_mm >= 0.5).astype(int))
        f1_05 = f1_score(gts, (probs_mm >= 0.5).astype(int), average="macro")
        auc = roc_auc_score(gts, probs_mm)

        best_t, best_t_acc = 0.5, acc_05
        for t in np.arange(0.30, 0.70, 0.01):
            cur_acc = accuracy_score(gts, (probs_mm >= t).astype(int))
            if cur_acc > best_t_acc:
                best_t_acc = cur_acc
                best_t = t

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d} ({elapsed:.1f}s): TrainLoss {total_loss / len(train_loader):.4f} | Acc@0.5: {acc_05*100:.2f}% | BestThreshAcc: {best_t_acc*100:.2f}% (T={best_t:.2f}) | F1: {f1_05*100:.2f}% | AUC: {auc:.4f}", flush=True)

        effective_acc = max(acc_05, best_t_acc)
        if effective_acc > best_dev_acc:
            best_dev_acc = effective_acc
            chosen_thresh = best_t if best_t_acc > acc_05 else 0.5
            chosen_preds = (probs_mm >= chosen_thresh).astype(int)
            best_payload = {
                "model_state_dict": {k: v.cpu().half() for k, v in model.state_dict().items()},
                "optimal_threshold": float(chosen_thresh),
                "best_val_f1": float(f1_score(gts, chosen_preds, average="macro")),
                "best_val_acc": float(effective_acc),
                "best_val_prec": float(precision_score(gts, chosen_preds, zero_division=0)),
                "best_val_rec": float(recall_score(gts, chosen_preds, zero_division=0)),
                "best_val_auc": float(auc),
                "dataset": "Archive (1).zip"
            }

    if best_payload is not None:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(best_payload, "checkpoints/best_hateful_memes.pt")
        print(f"\n[SUCCESS] Saved best model to checkpoints/best_hateful_memes.pt with Acc: {best_dev_acc*100:.2f}% (T={best_payload['optimal_threshold']:.2f})", flush=True)

if __name__ == "__main__":
    main()

