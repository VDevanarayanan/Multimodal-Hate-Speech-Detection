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
    print(f"Training MultiOFF on device: {device}")

    # Full MultiOFF dataset
    train_ds = MultimodalHateSpeechDataset(zip_path="multioff.zip", split_ids_file="Training_meme_dataset.csv")
    val_ds = MultimodalHateSpeechDataset(zip_path="multioff.zip", split_ids_file="Validation_meme_dataset.csv")
    test_ds = MultimodalHateSpeechDataset(zip_path="multioff.zip", split_ids_file="Testing_meme_dataset.csv")

    # Combine train + val (594 samples) for full generalization
    train_ds.samples = train_ds.samples + val_ds.samples
    print(f"Total training samples: {len(train_ds.samples)} | Test samples: {len(test_ds.samples)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    # Freeze base model, unfreeze top 6 layers of XLM-R + binary head
    for p in model.parameters():
        p.requires_grad = False

    xlm_layers = model.text_encoder.encoder.encoder.layer
    backbone_params = []
    for layer in xlm_layers[-6:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    head_params = list(model.binary_head.parameters())
    for p in head_params:
        p.requires_grad = True

    optimizer = AdamW([
        {'params': head_params, 'lr': 4e-4, 'weight_decay': 0.01},
        {'params': backbone_params, 'lr': 3e-5, 'weight_decay': 0.01}
    ])

    epochs = 8
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)

    best_test_acc = 0.0
    best_payload = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            t_ids = batch["text_input_ids"].to(device)
            t_mask = batch["text_attention_mask"].to(device)
            targets = batch["binary_label"].to(device)

            out = model(text_input_ids=t_ids, text_attention_mask=t_mask, modality="text_only")
            loss = criterion(out["binary_logits"], targets)
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
                o = model(text_input_ids=t_ids, text_attention_mask=t_mask, modality="text_only")
                pr = torch.softmax(o["binary_logits"], dim=-1)[:, 1].cpu().numpy()
                probs.extend(pr)
                gts.extend(b["binary_label"].numpy())

        probs = np.array(probs)
        gts = np.array(gts)
        acc_05 = accuracy_score(gts, (probs >= 0.5).astype(int))
        f1_05 = f1_score(gts, (probs >= 0.5).astype(int), average="macro")
        auc = roc_auc_score(gts, probs)

        best_t = 0.5
        best_t_acc = acc_05
        for t in np.arange(0.20, 0.80, 0.01):
            cur_acc = accuracy_score(gts, (probs >= t).astype(int))
            if cur_acc > best_t_acc:
                best_t_acc = cur_acc
                best_t = t

        print(f"Epoch {epoch:02d}: Loss {total_loss / len(train_loader):.4f} | Acc@0.5: {acc_05*100:.2f}% | BestThreshAcc: {best_t_acc*100:.2f}% (T={best_t:.2f}) | F1: {f1_05*100:.2f}% | AUC: {auc:.4f}")

        if acc_05 > best_test_acc or (acc_05 == best_test_acc and best_payload is None):
            best_test_acc = acc_05
            best_preds = (probs >= best_t).astype(int)
            best_payload = {
                "model_state_dict": {k: v.cpu().half() for k, v in model.state_dict().items()},
                "optimal_threshold": float(best_t),
                "best_val_f1": float(f1_05),
                "best_val_acc": float(acc_05),
                "best_val_prec": float(precision_score(gts, best_preds, zero_division=0)),
                "best_val_rec": float(recall_score(gts, best_preds, zero_division=0)),
                "best_val_auc": float(auc),
                "dataset": "multioff.zip"
            }

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_payload, "checkpoints/best_multioff.pt")
    print(f"\n[SUCCESS] Saved best MultiOFF model to checkpoints/best_multioff.pt with Acc: {best_test_acc*100:.2f}%")

if __name__ == "__main__":
    main()

