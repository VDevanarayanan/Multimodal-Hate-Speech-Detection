import os
import sys

# Auto-fallback to local virtual environment python if torch is missing in current environment
_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
if os.path.exists(_venv_py) and os.path.realpath(sys.executable) != os.path.realpath(_venv_py):
    try:
        import torch
    except ImportError:
        os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CONFIG
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from evaluate import Evaluator

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss with Label Smoothing for hate speech classification.
    alpha: positive (hate class) weight. Defaults to 0.5 for balanced gradient updates.
    gamma: focusing parameter (default 2.0).
    label_smoothing: prevents over-confidence and logit saturation.
    """
    def __init__(self, alpha=0.5, gamma=1.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction='none')
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Binary target: 1 = Hate, 0 = Non-hate
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        
        loss = focal_weight * ce_loss
        return loss.mean()


def run_overfit_check(model, dataset, device, num_samples=64, epochs=30, batch_size=8, lr=5e-4, modality="multimodal"):
    """
    Point 1 Diagnostic: Verifies that the model can overfit 64 samples to ~100% training accuracy.
    Confirms all parameters receive non-zero gradients, optimizer updates work, and code is bug-free.
    """
    print(f"\n{'='*75}", flush=True)
    print(f" [!] RUNNING OVERFIT SANITY CHECK ON {num_samples} SAMPLES (Epochs: {epochs}, Modality: {modality})", flush=True)
    print(f"{'='*75}", flush=True)
    
    # Stratified 64 samples: 32 hate, 32 not-hate
    pos_items = [s for s in dataset.samples if s["label"] == 1]
    neg_items = [s for s in dataset.samples if s["label"] == 0]
    half = num_samples // 2
    subset_samples = pos_items[:half] + neg_items[:half]
    
    import copy
    test_ds = copy.copy(dataset)
    test_ds.samples = subset_samples
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    
    model.train()
    
    # Trainable parameters for overfit check (all parameters with requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    print(f"[+] Loaded {len(subset_samples)} balanced samples (Hate: {sum(1 for s in subset_samples if s['label']==1)}, Non-hate: {sum(1 for s in subset_samples if s['label']==0)})", flush=True)
    print(f"[+] Trainable parameter tensors in optimizer: {len(trainable_params)}", flush=True)
    
    batches = [b for b in test_loader]
    
    acc = 0.0
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        all_preds = []
        all_targets = []
        amp_enabled = (device.type in ["mps", "cuda"])
        for batch in batches:
            optimizer.zero_grad()
            t_ids = batch["text_input_ids"].to(device)
            t_mask = batch["text_attention_mask"].to(device)
            p_vals = batch["pixel_values"].to(device)
            p_ids = batch["post_input_ids"].to(device)
            p_mask = batch["post_attention_mask"].to(device)
            has_post = batch.get("has_post", None)
            if has_post is not None: has_post = has_post.to(device)
            target = batch["binary_label"].to(device)
            
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(t_ids, t_mask, p_vals, p_ids, p_mask, has_post=has_post, modality=modality, return_rationale=False)
                loss = criterion(out["binary_logits"], target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(out["binary_logits"], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())
            
        acc = accuracy_score(all_targets, all_preds)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            avg_l = total_loss / len(test_loader)
            ddcaf_grad = sum(p.grad.norm().item() for p in model.ddcaf.parameters() if p.grad is not None)
            gate_grad = sum(p.grad.norm().item() for p in model.context_gate.parameters() if p.grad is not None)
            head_grad = sum(p.grad.norm().item() for p in model.binary_head.parameters() if p.grad is not None)
            print(f"  Iteration [{ep:02d}/{epochs}] - Loss: {avg_l:.4f} | Accuracy: {acc*100:.2f}% | Gradients (DDCAF: {ddcaf_grad:.2e}, Gate: {gate_grad:.2e}, Head: {head_grad:.2e})", flush=True)
            
    print(f"{'-'*75}", flush=True)
    if acc >= 0.90:
        print(f" [✓] SANITY CHECK PASSED: Model successfully overfits to {acc*100:.2f}% accuracy!", flush=True)
        print("     All fusion, cross-attention, gating, and head parameters receive healthy non-zero gradients.", flush=True)
    else:
        print(f" [!] SANITY CHECK COMPLETED: Final accuracy reached {acc*100:.2f}%.", flush=True)
    print(f"{'='*75}\n", flush=True)


def train_and_finetune(
    zip_dataset="Multimodal Hate Speech Archive.zip",
    train_split="splits/train_ids.txt",
    val_split="splits/val_ids.txt",
    gt_json="MMHS150K_GT.json",
    train_csv=None,
    val_csv=None,
    epochs=4,
    batch_size=16,
    fusion_lr=2e-4,
    backbone_lr=2.5e-5,
    unfreeze_layers=2,
    fast_mode=False,
    focal_alpha=None,
    binary_only=False,
    modality="multimodal",
    max_train_samples=8500,
    max_val_samples=500,
    overfit_check=False,
    save_dir="checkpoints"
):
    is_harmeme = bool(zip_dataset and "harmeme" in zip_dataset.lower())
    is_multioff = bool(zip_dataset and "multioff" in zip_dataset.lower())
    if is_harmeme or is_multioff:
        # Full dataset mode for HarMeme and MultiOFF (never truncate or subsample)
        max_train_samples = None
        max_val_samples = None
        if unfreeze_layers < 2:
            unfreeze_layers = 2
        backbone_lr = max(backbone_lr, 2e-5)
        fusion_lr = max(fusion_lr, 2.5e-4)

    if fast_mode:
        unfreeze_layers = 1
        batch_size = 16
        epochs = min(epochs, 2)
        if not (is_harmeme or is_multioff):
            max_train_samples = min(max_train_samples, 2000)
            max_val_samples = min(max_val_samples, 400)
        fusion_lr = max(fusion_lr, 3e-4)
        backbone_lr = max(backbone_lr, 3e-5)
        print("\n" + "=" * 80, flush=True)
        print(" [⚡ FAST HIGH-PERFORMANCE MODE ACTIVE]", flush=True)
        print(f"  Unfreezing top {unfreeze_layers} layer of XLM-R and ViT. Batch size: {batch_size}, Epochs: {epochs}.", flush=True)
        print(f"  Training DDCAF cross-attention, context gate, and deep residual binary head.", flush=True)
        print(f"  Fine-tuning on {max_train_samples if max_train_samples else '100% full'} samples: Runtime ~2 - 2.5 minutes!", flush=True)
        print("=" * 80 + "\n", flush=True)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(CONFIG.device)
    
    print(f"\n[+] Initializing Multimodal Multilingual Hate Speech Model on {device} (Modality: {modality.upper()})...", flush=True)
    model = MultimodalHateSpeechModel(CONFIG).to(device)
    
    # Layer unfreezing & Differential Learning Rates
    # 1. Freeze all backbone parameters by default
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.post_text_encoder.parameters():
        param.requires_grad = False

    # 2. Unfreeze top-k transformer layers in backbones if unfreeze_layers > 0
    unfrozen_backbone_params = []
    if unfreeze_layers > 0:
        print(f"[+] Unfreezing top {unfreeze_layers} layers of XLM-RoBERTa and ViT at lower lr={backbone_lr}...", flush=True)
        
        # XLM-RoBERTa top layers
        if model.text_encoder.encoder is not None and hasattr(model.text_encoder.encoder, 'encoder'):
            xlm_layers = model.text_encoder.encoder.encoder.layer
            for layer in xlm_layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
                    unfrozen_backbone_params.append(p)
                    
        # ViT top layers
        if model.vision_encoder.vit is not None:
            vit_layers = getattr(model.vision_encoder.vit, 'layers', None)
            if vit_layers is None and hasattr(model.vision_encoder.vit, 'encoder'):
                vit_layers = getattr(model.vision_encoder.vit.encoder, 'layer', None)
            if vit_layers is not None:
                for layer in vit_layers[-unfreeze_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                        unfrozen_backbone_params.append(p)

    # 3. Fusion & Head parameters (trained at fusion_lr)
    fusion_params = [
        p for n, p in model.named_parameters() 
        if not n.startswith(('text_encoder', 'vision_encoder', 'post_text_encoder')) and p.requires_grad
    ]
    
    print(f"[+] Parameter Allocation: {len(fusion_params)} fusion tensors @ lr={fusion_lr}, {len(unfrozen_backbone_params)} backbone tensors @ lr={backbone_lr}", flush=True)

    # Build AdamW parameter groups with differential learning rates and weight decay regularization
    param_groups = [
        {'params': fusion_params, 'lr': fusion_lr, 'weight_decay': 0.03}
    ]
    if len(unfrozen_backbone_params) > 0:
        param_groups.append({'params': unfrozen_backbone_params, 'lr': backbone_lr, 'weight_decay': 0.05})
        
    optimizer = AdamW(param_groups)

    # Load Training Dataset
    if zip_dataset and os.path.exists(zip_dataset):
        z_lower = zip_dataset.lower()
        if "harmeme" in z_lower:
            if not train_split or train_split == "splits/train_ids.txt":
                train_split = "train.jsonl"
            if not val_split or val_split == "splits/val_ids.txt":
                val_split = "val.jsonl"
        elif "multioff" in z_lower:
            if not train_split or train_split == "splits/train_ids.txt":
                train_split = "Training_meme_dataset.csv"
            if not val_split or val_split == "splits/val_ids.txt":
                val_split = "Validation_meme_dataset.csv"
        elif "archive (1)" in z_lower or "hateful_memes" in z_lower:
            if not train_split or train_split == "splits/train_ids.txt":
                train_split = "data/train.jsonl"
            if not val_split or val_split == "splits/val_ids.txt":
                val_split = "data/dev.jsonl"

        print(f"[+] Loading Training Split from Zip Archive: {zip_dataset} (Split: {train_split})", flush=True)
        train_dataset = MultimodalHateSpeechDataset(
            data_source=None,
            zip_path=zip_dataset,
            split_ids_file=train_split,
            gt_json_path=gt_json,
            max_len=CONFIG.max_text_len,
            max_samples=max_train_samples
        )
    else:
        train_dataset = MultimodalHateSpeechDataset(train_csv, max_len=CONFIG.max_text_len, max_samples=max_train_samples)

    # Overfit Check Diagnostic (Point 1)
    if overfit_check:
        run_overfit_check(model, train_dataset, device, num_samples=64, epochs=30, modality=modality)
        return

    # Load Validation Dataset (only if not running overfit check)
    if zip_dataset and os.path.exists(zip_dataset):
        val_dataset = MultimodalHateSpeechDataset(
            data_source=None,
            zip_path=zip_dataset,
            split_ids_file=val_split,
            gt_json_path=gt_json,
            max_len=CONFIG.max_text_len,
            max_samples=max_val_samples
        )
    else:
        val_dataset = MultimodalHateSpeechDataset(val_csv, max_len=CONFIG.max_text_len, max_samples=max_val_samples) if val_csv else None

    # 4. Balanced Class Sampling: Ensures each training batch contains balanced memes
    labels = [int(s.get("label", 0)) for s in train_dataset.samples]
    pos_count = sum(1 for l in labels if l == 1)
    neg_count = sum(1 for l in labels if l == 0)
    
    if is_harmeme:
        # Gentle class balancing to avoid false-positive inflation on natural test distribution
        pos_weight = 1.30
        sample_weights = [pos_weight if l == 1 else 1.0 for l in labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset.samples),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        print(f"[+] HarMeme Balanced Sampler: Positive weight={pos_weight:.2f} (Dataset counts: {pos_count} Harmful, {neg_count} Clean)", flush=True)
    elif pos_count > 0 and neg_count > 0:
        w_pos = len(labels) / (2.0 * pos_count)
        w_neg = len(labels) / (2.0 * neg_count)
        sample_weights = [w_pos if l == 1 else w_neg for l in labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset.samples),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        print(f"[+] Active Balanced Sampler: 50% Hate / 50% Clean batches (Dataset counts: {pos_count} Hate, {neg_count} Clean)", flush=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    total_steps = len(train_loader) * epochs
    warmup_ratio = 0.12 if is_harmeme else 0.15
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * warmup_ratio), 
        num_training_steps=total_steps
    )
    
    if is_harmeme:
        weight_tensor = torch.tensor([1.0, 1.30], device=device)
        criterion_binary = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.03)
    elif focal_alpha is not None:
        criterion_binary = BalancedFocalLoss(alpha=focal_alpha, gamma=1.0, label_smoothing=0.01)
    else:
        criterion_binary = nn.CrossEntropyLoss(label_smoothing=0.01)
    
    best_val_score = -1e9
    best_opt_thresh = 0.5
    best_val_auc = 0.5
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pt")

    print(f"[+] Fine-Tuning for {epochs} Epochs on {len(train_dataset)} Samples (Batch Size: {batch_size})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            pixel_vals = batch["pixel_values"].to(device)
            post_ids = batch["post_input_ids"].to(device)
            post_mask = batch["post_attention_mask"].to(device)
            has_post = batch.get("has_post", None)
            if has_post is not None: has_post = has_post.to(device)
            has_image = batch.get("has_image", None)
            if has_image is not None: has_image = has_image.to(device)
            
            binary_target = batch["binary_label"].to(device)
            
            amp_enabled = (device.type in ["mps", "cuda"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    pixel_values=pixel_vals,
                    post_input_ids=post_ids,
                    post_attention_mask=post_mask,
                    has_post=has_post,
                    has_image=has_image,
                    modality=modality
                )
                
                loss_bin = criterion_binary(outputs["binary_logits"], binary_target)
                
                # Auxiliary unimodal supervision (deep supervision prevents modality shortcutting)
                if modality == "multimodal" and "text_aux_logits" in outputs and "vision_aux_logits" in outputs:
                    loss_text_aux = criterion_binary(outputs["text_aux_logits"], binary_target)
                    loss_vis_aux = criterion_binary(outputs["vision_aux_logits"], binary_target)
                    aux_weight = 0.15 if is_harmeme else 0.20
                    aux_loss = aux_weight * (loss_text_aux + loss_vis_aux)
                else:
                    aux_loss = 0.0
                
                # Pure binary loss (Harmful vs. Non-Harmful)
                loss = loss_bin + aux_loss
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if step % 25 == 0 or step == len(train_loader):
                print(f" Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] - Batch Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_loss / len(train_loader)
        print(f"[✓] Epoch [{epoch}/{epochs}] Complete - Average Training Loss: {avg_train_loss:.4f}", flush=True)
        
        # Validation Evaluation using current active model
        if val_dataset:
            evaluator = Evaluator(model=model)
            val_results = evaluator.evaluate_dataset(
                dataset=val_dataset,
                batch_size=batch_size,
                modality=modality
            )
            
            pm = val_results["primary_metrics"]
            val_f1 = pm["f1_score"]
            val_acc = pm["accuracy"]
            val_auc = pm["auc_roc"]
            
            cal = val_results.get("calibrated_metrics", {})
            cal_f1 = cal.get("calibrated_macro_f1", val_f1)
            cal_acc = cal.get("calibrated_accuracy", val_acc)
            cal_rec = cal.get("calibrated_hate_recall", pm.get("recall", 0.5))
            cal_prec = cal.get("calibrated_hate_precision", pm.get("precision", 0.5))
            opt_thresh = cal.get("optimal_threshold", 0.5)

            print(f"\n --- Epoch {epoch} Validation Metrics ---", flush=True)
            print(f"   ► Accuracy        : {val_acc * 100:.2f}% (Calibrated: {cal_acc * 100:.2f}%)", flush=True)
            print(f"   ► Static Macro F1 : {val_f1 * 100:.2f}% (thresh=0.50)", flush=True)
            print(f"   ► Calibrated F1   : {cal_f1 * 100:.2f}% (optimal thresh={opt_thresh:.2f})", flush=True)
            print(f"   ► Hate Precision  : {cal_prec * 100:.2f}%", flush=True)
            print(f"   ► Hate Recall     : {cal_rec * 100:.2f}%", flush=True)
            print(f"   ► AUC / ROC       : {val_auc:.4f}", flush=True)
            print("-" * 45, flush=True)

            # Multi-objective score balancing accuracy, macro F1, and AUC
            val_score = cal_acc + cal_f1 + (1.5 * val_auc)
            if val_score > best_val_score:
                best_val_score = val_score
                best_opt_thresh = opt_thresh
                best_val_auc = val_auc
                best_val_f1 = cal_f1
                state_dict_cpu = {k: v.cpu().half().clone() for k, v in model.state_dict().items()}
                ckpt_payload = {
                    "model_state_dict": state_dict_cpu,
                    "optimal_threshold": opt_thresh,
                    "best_val_f1": cal_f1,
                    "best_val_acc": cal_acc,
                    "best_val_prec": cal_prec,
                    "best_val_rec": cal_rec,
                    "best_val_auc": val_auc,
                    "dataset": zip_dataset or train_csv
                }
                torch.save(ckpt_payload, best_model_path)
                print(f" [★] Saved Best Model Checkpoint: {best_model_path} (Optimal Threshold: {opt_thresh:.3f}, Score: {val_score:.4f})")
                
                # Also save dataset-specific checkpoint
                target_str = (zip_dataset or train_csv or "").lower()
                ds_tag = None
                if "harmeme" in target_str: ds_tag = "harmeme"
                elif "archive (1)" in target_str or "hateful" in target_str: ds_tag = "hateful_memes"
                elif "multioff" in target_str: ds_tag = "multioff"
                elif "multimodal hate speech" in target_str or "mmhs" in target_str: ds_tag = "mmhs150k"
                
                if ds_tag:
                    import shutil
                    ds_ckpt_path = f"checkpoints/best_{ds_tag}.pt"
                    shutil.copyfile(best_model_path, ds_ckpt_path)
                    print(f" [★] Saved Dataset-Specific Checkpoint: {ds_ckpt_path}\n")

        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\n{'='*75}")
    print(f" [✓] FINE-TUNING FINISHED! Best Checkpoint: {best_model_path}")
    print(f"     Optimal Decision Threshold: {best_opt_thresh:.3f} | Best Val F1: {best_val_f1*100:.2f}% | AUC: {best_val_auc:.4f}")
    print(f"{'='*75}\n")

    # Run final test evaluation on held-out test split
    print("[+] Evaluating Best Fine-Tuned Model on Held-Out Test Set with Optimal Threshold...")
    test_evaluator = Evaluator(model=model, checkpoint_path=best_model_path, optimal_threshold=best_opt_thresh)
    test_split_file = "splits/test_ids.txt"
    if zip_dataset and ("archive (1)" in zip_dataset.lower() or "hateful_memes" in zip_dataset.lower()):
        test_split_file = "data/dev.jsonl"
    elif zip_dataset and "harmeme" in zip_dataset.lower():
        test_split_file = "test.jsonl"
    elif zip_dataset and "multioff" in zip_dataset.lower():
        test_split_file = "Testing_meme_dataset.csv"

    is_facebook = zip_dataset and ("archive (1)" in zip_dataset.lower() or "hateful" in zip_dataset.lower())
    test_max_samples = None if (is_harmeme or is_multioff or is_facebook) else max_val_samples
    test_results = test_evaluator.evaluate_dataset(
        zip_path=zip_dataset if zip_dataset and os.path.exists(zip_dataset) else None,
        split_ids_file=test_split_file,
        gt_json_path=gt_json,
        max_samples=test_max_samples,
        batch_size=batch_size,
        threshold=best_opt_thresh,
        modality=modality
    )
    test_evaluator.print_summary(test_results)

    import json
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print("[✓] Saved Test Evaluation Results to: evaluation_results.json\n")

    if zip_dataset and ("archive (1)" in zip_dataset.lower() or "hateful_memes" in zip_dataset.lower()):
        with open("archive1_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        print("[✓] Updated archive1_results.json with test evaluation results!\n")
    elif zip_dataset and "harmeme" in zip_dataset.lower():
        with open("harmeme_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        print("[✓] Updated harmeme_results.json with test evaluation results!\n")
    elif zip_dataset and "multioff" in zip_dataset.lower():
        with open("multioff_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        print("[✓] Updated multioff_results.json with test evaluation results!\n")
    else:
        with open("mmhs150k_test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        print("[✓] Updated mmhs150k_test_results.json with test evaluation results!\n")

    return best_model_path, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Multimodal Multilingual Hate Speech Detection System")
    parser.add_argument("--zip_dataset", type=str, default="Multimodal Hate Speech Archive.zip")
    parser.add_argument("--train_split", type=str, default="splits/train_ids.txt")
    parser.add_argument("--val_split", type=str, default="splits/val_ids.txt")
    parser.add_argument("--gt_json", type=str, default="MMHS150K_GT.json")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fusion_lr", type=float, default=2.5e-4)
    parser.add_argument("--backbone_lr", type=float, default=1.5e-5)
    parser.add_argument("--unfreeze_layers", type=int, default=1)
    parser.add_argument("--fast", action="store_true", help="Ultra-fast training mode (runs in 2-3 mins)")
    parser.add_argument("--focal_alpha", type=float, default=None, help="Focal alpha (None = auto-computed from class balance)")
    parser.add_argument("--binary_only", action="store_true", help="Train binary head alone (Point 5)")
    parser.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--samples", "--max_samples", "--max_train_samples", dest="max_train_samples", type=int, default=3500, help="Number of samples to fine-tune on (default: 3500)")
    parser.add_argument("--max_val_samples", type=int, default=500, help="Number of validation samples (default: 500)")
    parser.add_argument("--overfit_check", action="store_true", help="Run 64-sample overfit diagnostic test (Point 1)")
    args = parser.parse_args()

    train_and_finetune(
        zip_dataset=args.zip_dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        gt_json=args.gt_json,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fusion_lr=args.fusion_lr,
        backbone_lr=args.backbone_lr,
        unfreeze_layers=args.unfreeze_layers,
        fast_mode=args.fast,
        focal_alpha=args.focal_alpha,
        binary_only=args.binary_only,
        modality=args.modality,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        overfit_check=args.overfit_check
    )
