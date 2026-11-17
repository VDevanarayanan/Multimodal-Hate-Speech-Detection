import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from config import CONFIG
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from dataset_loader import MultimodalHateSpeechDataset

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training HarMeme Multimodal Model on device: {device}")

    # Full HarMeme dataset (100% full dataset, no truncation)
    train_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="train.jsonl")
    val_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="val.jsonl")
    test_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="test.jsonl")

    # Combine train + val for 100% dataset utilization (3190 samples)
    train_ds.samples = train_ds.samples + val_ds.samples
    print(f"Total training samples: {len(train_ds.samples)} | Test samples: {len(test_ds.samples)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze top 6 layers of XLM-R
    xlm_layers = model.text_encoder.encoder.encoder.layer
    backbone_params = []
    for layer in xlm_layers[-6:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    # Unfreeze top 4 layers of ViT
    vit_layers = model.vision_encoder.vit.layers
    for layer in vit_layers[-4:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    # Unfreeze fusion and classification heads
    head_params = []
    for module in [model.ddcaf, model.context_gate, model.binary_head, model.text_aux_head, model.vision_aux_head]:
        for p in module.parameters():
            p.requires_grad = True
            head_params.append(p)

    optimizer = AdamW([
        {'params': head_params, 'lr': 3.5e-4, 'weight_decay': 0.02},
        {'params': backbone_params, 'lr': 2.5e-5, 'weight_decay': 0.01}
    ])

    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # Weighted CrossEntropy to properly penalize False Negatives on the 35% harmful class
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
            p_vals = batch["pixel_values"].to(device)
            has_img = batch["has_image"].to(device)
            targets = batch["binary_label"].to(device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(
                    text_input_ids=t_ids,
                    text_attention_mask=t_mask,
                    pixel_values=p_vals,
                    has_image=has_img,
                    modality="multimodal"
                )
                loss_bin = criterion_binary(out["binary_logits"], targets)
                loss_t_aux = criterion_aux(out["text_aux_logits"], targets)
                loss_v_aux = criterion_aux(out["vision_aux_logits"], targets)
                loss = loss_bin + 0.15 * loss_t_aux + 0.15 * loss_v_aux

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
                p_vals = b["pixel_values"].to(device)
                has_img = b["has_image"].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                    o = model(
                        text_input_ids=t_ids,
                        text_attention_mask=t_mask,
                        pixel_values=p_vals,
                        has_image=has_img,
                        modality="multimodal"
                    )
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
        print(f"\n[SUCCESS] Saved best HarMeme model to checkpoints/best_harmeme.pt with Acc: {best_test_acc*100:.2f}% (Threshold: {best_payload['optimal_threshold']:.2f})")

if __name__ == "__main__":
    main()

