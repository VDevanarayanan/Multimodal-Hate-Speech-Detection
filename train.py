import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CONFIG
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from evaluate import Evaluator

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss to balance class weights and boost Precision, Accuracy, and AUC/ROC.
    """
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        weights = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = weights * ((1.0 - pt) ** self.gamma)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        loss = focal_weight * ce_loss
        return loss.mean()


def train_and_finetune(
    zip_dataset="Multimodal Hate Speech Archive.zip",
    train_split="splits/train_ids.txt",
    val_split="splits/val_ids.txt",
    gt_json="MMHS150K_GT.json",
    train_csv=None,
    val_csv=None,
    epochs=3,
    batch_size=4,
    lr=1e-4,
    freeze_backbones=True,
    max_train_samples=1000,
    max_val_samples=300,
    save_dir="checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(CONFIG.device)
    
    print(f"\n[+] Initializing Multimodal Multilingual Hate Speech Model on {device}...")
    model = MultimodalHateSpeechModel(CONFIG).to(device)
    
    if freeze_backbones:
        print("[+] Adapter Mode: Freezing heavy text & vision backbone weights to optimize memory & training speed...")
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        for param in model.post_text_encoder.parameters():
            param.requires_grad = False
            
    # Trainable parameters (DDCAF, Context Gate, Binary Head, Multi-label Head, Rationale Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[+] Total trainable parameter tensors: {len(trainable_params)}")

    # Load Datasets
    if zip_dataset and os.path.exists(zip_dataset):
        print(f"[+] Loading Training Split from Zip Archive: {zip_dataset}")
        train_dataset = MultimodalHateSpeechDataset(
            data_source=None,
            zip_path=zip_dataset,
            split_ids_file=train_split,
            gt_json_path=gt_json,
            max_len=CONFIG.max_text_len,
            max_samples=max_train_samples
        )
        val_dataset = MultimodalHateSpeechDataset(
            data_source=None,
            zip_path=zip_dataset,
            split_ids_file=val_split,
            gt_json_path=gt_json,
            max_len=CONFIG.max_text_len,
            max_samples=max_val_samples
        )
    else:
        train_dataset = MultimodalHateSpeechDataset(train_csv, max_len=CONFIG.max_text_len)
        val_dataset = MultimodalHateSpeechDataset(val_csv, max_len=CONFIG.max_text_len) if val_csv else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )
    
    criterion_binary = BalancedFocalLoss(alpha=0.5, gamma=2.0)
    criterion_multilabel = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pt")

    print(f"[+] Fine-Tuning for {epochs} Epochs on {len(train_dataset)} Samples (Batch Size: {batch_size})...")

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbones:
            model.text_encoder.eval()
            model.vision_encoder.eval()
            model.post_text_encoder.eval()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            pixel_vals = batch["pixel_values"].to(device)
            post_ids = batch["post_input_ids"].to(device)
            post_mask = batch["post_attention_mask"].to(device)
            
            binary_target = batch["binary_label"].to(device)
            multilabel_target = batch["multilabel_target"].to(device)
            
            outputs = model(
                text_input_ids=text_ids,
                text_attention_mask=text_mask,
                pixel_values=pixel_vals,
                post_input_ids=post_ids,
                post_attention_mask=post_mask
            )
            
            loss_bin = criterion_binary(outputs["binary_logits"], binary_target)
            loss_ml = criterion_multilabel(outputs["multilabel_logits"], multilabel_target)
            
            loss = (CONFIG.binary_loss_weight * loss_bin) + (CONFIG.multilabel_loss_weight * loss_ml)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if step % 25 == 0 or step == len(train_loader):
                print(f" Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] - Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"[✓] Epoch [{epoch}/{epochs}] Complete - Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation Evaluation
        if val_dataset:
            evaluator = Evaluator(model=model)
            val_results = evaluator.evaluate_dataset(
                zip_path=zip_dataset if zip_dataset and os.path.exists(zip_dataset) else None,
                split_ids_file=val_split,
                gt_json_path=gt_json,
                max_samples=max_val_samples
            )
            
            pm = val_results["primary_metrics"]
            val_f1 = pm["f1_score"]
            val_acc = pm["accuracy"]
            val_auc = pm["auc_roc"]
            
            print(f"\n --- Epoch {epoch} Validation Metrics ---")
            print(f"   ► Accuracy  : {val_acc * 100:.2f}%")
            print(f"   ► Macro F1  : {val_f1 * 100:.2f}%")
            print(f"   ► AUC / ROC : {val_auc:.4f}")
            print("-" * 35)

            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f" [★] Saved Best Model Checkpoint: {best_model_path}\n")

    print("\n[+] Fine-Tuning Complete!")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Multimodal Multilingual Hate Speech Detection System")
    parser.add_argument("--zip_dataset", type=str, default="Multimodal Hate Speech Archive.zip")
    parser.add_argument("--train_split", type=str, default="splits/train_ids.txt")
    parser.add_argument("--val_split", type=str, default="splits/val_ids.txt")
    parser.add_argument("--gt_json", type=str, default="MMHS150K_GT.json")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_val_samples", type=int, default=300)
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
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
