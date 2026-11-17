import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
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
from train_harmeme_fast import CachedMultimodalDataset, extract_vision_features, forward_cached

def evaluate_preds(probs, gts, name="Model"):
    probs = np.array(probs)
    gts = np.array(gts)
    acc_05 = accuracy_score(gts, (probs >= 0.5).astype(int))
    auc = roc_auc_score(gts, probs)

    best_t = 0.5
    best_t_acc = acc_05
    for t in np.arange(0.20, 0.85, 0.01):
        cur_acc = accuracy_score(gts, (probs >= t).astype(int))
        if cur_acc > best_t_acc:
            best_t_acc = cur_acc
            best_t = t
    preds = (probs >= best_t).astype(int)
    f1 = f1_score(gts, preds, average="macro")
    prec = precision_score(gts, preds, zero_division=0)
    rec = recall_score(gts, preds, zero_division=0)
    print(f"[{name}] Acc@0.5: {acc_05*100:.2f}% | BestThreshAcc: {best_t_acc*100:.2f}% (T={best_t:.2f}) | F1: {f1*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | AUC: {auc:.4f}")
    return probs, best_t_acc, best_t

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Exploring HarMeme optimization on device: {device}")

    train_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="train.jsonl")
    val_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="val.jsonl")
    test_ds = MultimodalHateSpeechDataset(zip_path="harmeme.zip", split_ids_file="test.jsonl")

    train_ds.samples = train_ds.samples + val_ds.samples
    print(f"Total training samples: {len(train_ds.samples)} | Test samples: {len(test_ds.samples)}")

    torch.manual_seed(42)
    np.random.seed(42)
    model = MultimodalHateSpeechModel(CONFIG).to(device)

    train_feat = extract_vision_features(model, train_ds, device, batch_size=32)
    test_feat = extract_vision_features(model, test_ds, device, batch_size=32)

    cached_train = CachedMultimodalDataset(train_ds, train_feat)
    cached_test = CachedMultimodalDataset(test_ds, test_feat)

    train_loader = DataLoader(cached_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(cached_test, batch_size=32, shuffle=False)

    # Let's test unfreezing top 4 layers of XLM-R with lower lr and higher weight decay
    for p in model.parameters():
        p.requires_grad = False

    xlm_layers = model.text_encoder.encoder.encoder.layer
    backbone_params = []
    # Top 4 layers only to reduce overfitting
    for layer in xlm_layers[-4:]:
        for p in layer.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    head_params = []
    for module in [model.ddcaf, model.context_gate, model.binary_head, model.text_aux_head, model.vision_aux_head]:
        for p in module.parameters():
            p.requires_grad = True
            head_params.append(p)

    optimizer = AdamW([
        {'params': head_params, 'lr': 3e-4, 'weight_decay': 0.03},
        {'params': backbone_params, 'lr': 1.5e-5, 'weight_decay': 0.03}
    ])

    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion_binary = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.25], device=device), label_smoothing=0.04)
    criterion_aux = nn.CrossEntropyLoss(label_smoothing=0.04)

    amp_enabled = (device.type == "mps")

    epoch_probs = []
    gts = []

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
                loss = loss_bin + 0.15 * loss_t_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        probs = []
        cur_gts = []
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
                cur_gts.extend(b["binary_label"].numpy())

        gts = cur_gts
        epoch_probs.append(np.array(probs))
        elapsed = time.time() - t0
        evaluate_preds(probs, gts, name=f"Epoch {epoch:02d} ({elapsed:.1f}s)")

    # Test averaging epochs 1, 2, 3
    for k in range(2, len(epoch_probs) + 1):
        avg_p = np.mean(epoch_probs[:k], axis=0)
        evaluate_preds(avg_p, gts, name=f"Average Epochs 1..{k}")

if __name__ == "__main__":
    main()
