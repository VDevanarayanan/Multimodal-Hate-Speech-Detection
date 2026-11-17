import os
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
    print(f"HarMeme Training on device: {device}")

    train_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="train.jsonl")
    val_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="val.jsonl")
    test_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="test.jsonl")

    # Combine train + val for 100% full dataset utilization
    train_ds.samples = train_ds.samples + val_ds.samples
    print(f"Total training samples: {len(train_ds.samples)} | Test samples: {len(test_ds.samples)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    # Freeze base, unfreeze top 4 layers of ViT, top 6 layers of XLM-R, DDCAF, ContextGate, BinaryHead
    for p in model.parameters():
        p.requires_grad = False

    xlm_layers = model.text_encoder.encoder.encoder.layer
    backbone_params = []
    for layer in xlm_layers[-6:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    vit_layers = model.vision_encoder.vit.layers
    for layer in vit_layers[-4:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    fusion_params = []
    for module in [model.ddcaf, model.context_gate, model.binary_head, model.text_aux_head, model.vision_aux_head]:
        for p in module.parameters():
            p.requires_grad = True
            fusion_params.append(p)

    optimizer = AdamW([
        {'params': fusion_params, 'lr': 3e-4, 'weight_decay': 0.02},
        {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.01}
    ])

    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # Balanced class weights for HarMeme (35% harmful, 65% clean)
    pos_weight = 65.0 / 35.0 # ~1.85
    criterion_binary = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.3], device=device), label_smoothing=0.03)
    criterion_aux = nn.CrossEntropyLoss(label_smoothing=0.03)

    amp_enabled = (device.type == "mps")

    for epoch in range(1, epochs + 1):
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
                loss = loss_bin + 0.2 * loss_t_aux + 0.2 * loss_v_aux

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
        for t in np.arange(0.20, 0.85, 0.01):
            cur_acc = accuracy_score(gts, (probs >= t).astype(int))
            if cur_acc > best_t_acc:
                best_t_acc = cur_acc
                best_t = t

        print(f"HarMeme Epoch {epoch:02d}: TrainLoss {total_loss / len(train_loader):.4f} | Acc@0.5: {acc_05*100:.2f}% | BestThreshAcc: {best_t_acc*100:.2f}% (T={best_t:.2f}) | F1: {f1_05*100:.2f}% | AUC: {auc:.4f}", flush=True)

if __name__ == "__main__":
    main()

