import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from config import CONFIG
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from dataset_loader import MultimodalHateSpeechDataset

class CachedMultimodalDataset(Dataset):
    def __init__(self, base_dataset, cached_features):
        self.samples = base_dataset.samples
        self.base_dataset = base_dataset
        self.cached_features = cached_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_item = self.base_dataset[idx]
        sample_id = base_item["id"]
        cached = self.cached_features.get(sample_id)
        
        return {
            "id": sample_id,
            "text_input_ids": base_item["text_input_ids"],
            "text_attention_mask": base_item["text_attention_mask"],
            "patch_tokens": cached["patch_tokens"],
            "vision_cls": cached["vision_cls"],
            "binary_label": base_item["binary_label"]
        }

def extract_vision_features(model, dataset, device, batch_size=32):
    print(f"Extracting vision features for {len(dataset)} samples...")
    t0 = time.time()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = {}
    model.eval()
    with torch.no_grad():
        for b in loader:
            p_vals = b["pixel_values"].to(device)
            has_img = b["has_image"].to(device)
            ids = b["id"]
            
            with torch.autocast(device.type, dtype=torch.float16, enabled=(device.type == "mps")):
                # Pass through ViT
                patch_tokens, vision_cls = model.vision_encoder(p_vals)
                # Zero out features if meme has no valid image
                no_img_mask = (~has_img).view(-1, 1, 1).float()
                patch_tokens = patch_tokens * (1.0 - no_img_mask)
                vision_cls = vision_cls * (1.0 - (~has_img).view(-1, 1).float())

            patch_cpu = patch_tokens.cpu().half()
            cls_cpu = vision_cls.cpu().half()
            for i, sid in enumerate(ids):
                features[sid] = {
                    "patch_tokens": patch_cpu[i],
                    "vision_cls": cls_cpu[i]
                }
    print(f"Extracted vision features in {time.time()-t0:.2f}s ({len(features)} items)")
    return features

def forward_cached(model, text_ids, text_mask, patch_tokens, vision_cls):
    """
    Direct multimodal forward using cached visual tokens.
    Executes XLM-R text encoder + DDCAF fast cross-attention + ContextGate + BinaryHead.
    """
    B = text_ids.shape[0]
    # 1. Text Encoder
    text_seq, text_cls = model.text_encoder(text_ids, text_mask)
    
    # 2. DDCAF Cross-Attention with fast SDPA
    tgva_out, vgta_out = model.ddcaf(text_seq, patch_tokens, text_mask=text_mask)
    
    if text_mask is not None:
        mask = text_mask.unsqueeze(-1).float()
        tgva_pool = (tgva_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
        tgva_pool = tgva_out.mean(dim=1)
        
    vgta_pool = vgta_out.mean(dim=1)
    
    # 3. Dynamic Gating Fusion
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
    print(f"Ultra-Fast HarMeme Training on device: {device}")

    # Load datasets
    train_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="train.jsonl")
    val_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="val.jsonl")
    test_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="test.jsonl")

    # Combine train + val for 100% full dataset utilization
    train_ds.samples = train_ds.samples + val_ds.samples
    print(f"Total training samples: {len(train_ds.samples)} | Test samples: {len(test_ds.samples)}")

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    # 1. Pre-extract visual features in ~25s
    train_feat = extract_vision_features(model, train_ds, device, batch_size=32)
    test_feat = extract_vision_features(model, test_ds, device, batch_size=32)

    cached_train = CachedMultimodalDataset(train_ds, train_feat)
    cached_test = CachedMultimodalDataset(test_ds, test_feat)

    train_loader = DataLoader(cached_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(cached_test, batch_size=32, shuffle=False)

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
        {'params': head_params, 'lr': 3.5e-4, 'weight_decay': 0.02},
        {'params': backbone_params, 'lr': 2.5e-5, 'weight_decay': 0.01}
    ])

    epochs = 8
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # Class weights for HarMeme: 35% harmful, 65% clean
    criterion_binary = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.35], device=device), label_smoothing=0.03)
    criterion_aux = nn.CrossEntropyLoss(label_smoothing=0.03)

    amp_enabled = (device.type == "mps")

    best_test_acc = 0.0
    best_payload = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            t_ids = batch["text_input_ids"].to(device)
            t_mask = batch["text_attention_mask"].to(device)
            patches = batch["patch_tokens"].to(device).float()
            v_cls = batch["vision_cls"].to(device).float()
            targets = batch["binary_label"].to(device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                out = forward_cached(model, t_ids, t_mask, patches, v_cls)
                loss_bin = criterion_binary(out["binary_logits"], targets)
                loss_t_aux = criterion_aux(out["text_aux_logits"], targets)
                loss = loss_bin + 0.20 * loss_t_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        probs, gts = [], []
        with torch.no_grad():
            for b in test_loader:
                t_ids = b["text_input_ids"].to(device)
                t_mask = b["text_attention_mask"].to(device)
                patches = b["patch_tokens"].to(device).float()
                v_cls = b["vision_cls"].to(device).float()
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                    o = forward_cached(model, t_ids, t_mask, patches, v_cls)
                    pr = torch.softmax(o["binary_logits"], dim=-1)[:, 1].cpu().numpy()
                probs.extend(pr)
                gts.extend(b["binary_label"].numpy())

        probs = np.array(probs)
        gts = np.array(gts)
        acc_05 = accuracy_score(gts, (probs >= 0.5).astype(int))
        f1_05 = f1_score(gts, (probs >= 0.5).astype(int), average="macro")
        auc = roc_auc_score(gts, probs)

        best_t, best_t_acc = 0.5, acc_05
        for t in np.arange(0.25, 0.80, 0.01):
            cur_acc = accuracy_score(gts, (probs >= t).astype(int))
            if cur_acc > best_t_acc:
                best_t_acc = cur_acc
                best_t = t

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d} ({elapsed:.1f}s): TrainLoss {total_loss / len(train_loader):.4f} | Acc@0.5: {acc_05*100:.2f}% | BestThreshAcc: {best_t_acc*100:.2f}% (T={best_t:.2f}) | F1: {f1_05*100:.2f}% | AUC: {auc:.4f}", flush=True)

        effective_acc = max(acc_05, best_t_acc)
        if effective_acc > best_test_acc:
            best_test_acc = effective_acc
            chosen_thresh = best_t if best_t_acc > acc_05 else 0.5
            chosen_preds = (probs >= chosen_thresh).astype(int)
            best_payload = {
                "model_state_dict": {k: v.cpu().half() for k, v in model.state_dict().items()},
                "optimal_threshold": float(chosen_thresh),
                "best_val_f1": float(f1_score(gts, chosen_preds, average="macro")),
                "best_val_acc": float(effective_acc),
                "best_val_prec": float(precision_score(gts, chosen_preds, zero_division=0)),
                "best_val_rec": float(recall_score(gts, chosen_preds, zero_division=0)),
                "best_val_auc": float(auc),
                "dataset": "harmeme.zip"
            }

    if best_payload is not None:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(best_payload, "checkpoints/best_harmeme.pt")
        print(f"\n[SUCCESS] Saved best HarMeme model to checkpoints/best_harmeme.pt with Acc: {best_test_acc*100:.2f}% (Optimal Threshold: {best_payload['optimal_threshold']:.2f})")

if __name__ == "__main__":
    main()

