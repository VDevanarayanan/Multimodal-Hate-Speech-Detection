import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from config import CONFIG
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from dataset_loader import MultimodalHateSpeechDataset

MMAP_DIR = "scratch/fb_mmap"

class MemmapMultimodalDataset(Dataset):
    def __init__(self, base_dataset, patches_mmap, cls_mmap):
        self.samples = base_dataset.samples
        self.base_dataset = base_dataset
        self.patches_mmap = patches_mmap
        self.cls_mmap = cls_mmap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_item = self.base_dataset[idx]
        patch_tok = torch.from_numpy(np.array(self.patches_mmap[idx])).float()
        vis_cls = torch.from_numpy(np.array(self.cls_mmap[idx])).float()
        
        return {
            "id": base_item["id"],
            "text_input_ids": base_item["text_input_ids"],
            "text_attention_mask": base_item["text_attention_mask"],
            "patch_tokens": patch_tok,
            "vision_cls": vis_cls,
            "binary_label": base_item["binary_label"]
        }

def extract_and_memmap(model, dataset, split_name, device, batch_size=32):
    os.makedirs(MMAP_DIR, exist_ok=True)
    N = len(dataset)
    patch_file = os.path.join(MMAP_DIR, f"{split_name}_patches.dat")
    cls_file = os.path.join(MMAP_DIR, f"{split_name}_cls.dat")

    if os.path.exists(patch_file) and os.path.exists(cls_file):
        print(f"[+] Found existing memmap cache for {split_name} ({N} samples), loading directly...", flush=True)
        patches_mmap = np.memmap(patch_file, dtype="float16", mode="r", shape=(N, 196, 768))
        cls_mmap = np.memmap(cls_file, dtype="float16", mode="r", shape=(N, 768))
        return patches_mmap, cls_mmap

    print(f"[+] Extracting vision features for {split_name} ({N} samples) to disk memmap...", flush=True)
    t0 = time.time()
    patches_mmap = np.memmap(patch_file, dtype="float16", mode="w+", shape=(N, 196, 768))
    cls_mmap = np.memmap(cls_file, dtype="float16", mode="w+", shape=(N, 768))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    idx = 0
    with torch.no_grad():
        for b in loader:
            p_vals = b["pixel_values"].to(device)
            has_img = b["has_image"].to(device)
            B = p_vals.shape[0]

            with torch.autocast(device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                patch_tokens, vision_cls = model.vision_encoder(p_vals)
                no_img_mask = (~has_img).view(-1, 1, 1).float()
                patch_tokens = patch_tokens * (1.0 - no_img_mask)
                vision_cls = vision_cls * (1.0 - (~has_img).view(-1, 1).float())

            patches_mmap[idx : idx + B] = patch_tokens.cpu().half().numpy()
            cls_mmap[idx : idx + B] = vision_cls.cpu().half().numpy()
            idx += B
            if idx % 1000 == 0 or idx == N:
                print(f"    Progress: {idx}/{N} images ({time.time()-t0:.1f}s)...", flush=True)

    patches_mmap.flush()
    cls_mmap.flush()
    elapsed = time.time() - t0
    print(f"[✓] Extracted & cached {N} images in {elapsed:.2f}s ({N/elapsed:.1f} img/s)", flush=True)

    # Re-open in read-only mode
    patches_mmap = np.memmap(patch_file, dtype="float16", mode="r", shape=(N, 196, 768))
    cls_mmap = np.memmap(cls_file, dtype="float16", mode="r", shape=(N, 768))
    return patches_mmap, cls_mmap

def forward_cached(model, text_ids, text_mask, patch_tokens, vision_cls):
    """
    Direct multimodal forward using cached visual tokens.
    Executes XLM-R text encoder + DDCAF fast cross-attention + ContextGate + BinaryHead.
    """
    text_seq, text_cls = model.text_encoder(text_ids, text_mask)
    
    tgva_out, vgta_out = model.ddcaf(text_seq, patch_tokens, text_mask=text_mask)
    
    if text_mask is not None:
        mask = text_mask.unsqueeze(-1).float()
        tgva_pool = (tgva_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
        tgva_pool = tgva_out.mean(dim=1)
        
    vgta_pool = vgta_out.mean(dim=1)
    
    fused_rep, gate_weights = model.context_gate(
        main_text_cls=text_cls,
        vision_cls=vision_cls,
        tgva_pool=tgva_pool,
        vgta_pool=vgta_pool,
        post_text_cls=None
    )
    
    binary_logits = model.binary_head(fused_rep)
    text_aux_logits = model.text_aux_head(text_cls)
    vision_aux_logits = model.vision_aux_head(vision_cls)
    
    return {
        "binary_logits": binary_logits,
        "text_aux_logits": text_aux_logits,
        "vision_aux_logits": vision_aux_logits,
        "gate_weights": gate_weights
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"===========================================================")
    print(f" Fast Facebook Hateful Memes Training on device: {device}")
    print(f"===========================================================")

    # 1. Load full datasets
    train_ds = MultimodalHateSpeechDataset(zip_path="Archive (1).zip", split_ids_file="data/train.jsonl")
    dev_ds = MultimodalHateSpeechDataset(zip_path="Archive (1).zip", split_ids_file="data/dev.jsonl")

    print(f"Total training samples: {len(train_ds.samples)} | Dev samples: {len(dev_ds.samples)}")

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    # 2. Extract visual features to disk memmap
    train_patches, train_cls = extract_and_memmap(model, train_ds, "train", device, batch_size=32)
    dev_patches, dev_cls = extract_and_memmap(model, dev_ds, "dev", device, batch_size=32)

    cached_train = MemmapMultimodalDataset(train_ds, train_patches, train_cls)
    cached_dev = MemmapMultimodalDataset(dev_ds, dev_patches, dev_cls)

    train_loader = DataLoader(cached_train, batch_size=16, shuffle=True)
    dev_loader = DataLoader(cached_dev, batch_size=32, shuffle=False)

    # Freeze base model, unfreeze top 6 layers of XLM-R + DDCAF + ContextGate + BinaryHead + Aux heads
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
        {'params': head_params, 'lr': 3.5e-4, 'weight_decay': 0.02},
        {'params': backbone_params, 'lr': 2.0e-5, 'weight_decay': 0.01}
    ])

    epochs = 5
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # In train: 5450 clean vs 3050 hate (ratio = 1.787). In dev: 250 clean vs 250 hate (50/50 balanced).
    criterion_binary = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.75], device=device), label_smoothing=0.03)
    criterion_aux = nn.CrossEntropyLoss(label_smoothing=0.03)

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
                loss_bin = criterion_binary(out["binary_logits"], targets)
                loss_t_aux = criterion_aux(out["text_aux_logits"], targets)
                loss_v_aux = criterion_aux(out["vision_aux_logits"], targets)
                loss = loss_bin + 0.15 * loss_t_aux + 0.05 * loss_v_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate on dev
        model.eval()
        probs_mm, probs_text, gts = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                t_ids = b["text_input_ids"].to(device)
                t_mask = b["text_attention_mask"].to(device)
                patches = b["patch_tokens"].to(device)
                v_cls = b["vision_cls"].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                    o = forward_cached(model, t_ids, t_mask, patches, v_cls)
                    pr_mm = torch.softmax(o["binary_logits"], dim=-1)[:, 1].cpu().numpy()
                    pr_txt = torch.softmax(o["text_aux_logits"], dim=-1)[:, 1].cpu().numpy()
                probs_mm.extend(pr_mm)
                probs_text.extend(pr_txt)
                gts.extend(b["binary_label"].numpy())

        probs_mm = np.array(probs_mm)
        probs_text = np.array(probs_text)
        gts = np.array(gts)

        # Evaluate multimodal probabilities
        acc_05 = accuracy_score(gts, (probs_mm >= 0.5).astype(int))
        f1_05 = f1_score(gts, (probs_mm >= 0.5).astype(int), average="macro")
        auc = roc_auc_score(gts, probs_mm)

        best_t, best_t_acc = 0.5, acc_05
        for t in np.arange(0.20, 0.80, 0.01):
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
        print(f"\n[SUCCESS] Saved best Facebook Hateful Memes model to checkpoints/best_hateful_memes.pt with Acc: {best_dev_acc*100:.2f}% (Optimal Threshold: {best_payload['optimal_threshold']:.2f})", flush=True)

if __name__ == "__main__":
    main()
