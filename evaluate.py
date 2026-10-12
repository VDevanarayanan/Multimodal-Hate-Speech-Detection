import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from config import CONFIG
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel

class Evaluator:
    """
    Evaluation Engine for Multimodal Multilingual Hate Speech Detection System.
    Calculates Accuracy, Precision, Recall, F1-Score, and AUC/ROC
    for any user-provided dataset (CSV, JSON, or Zip archive).
    """
    def __init__(self, model=None, checkpoint_path=None, config=CONFIG):
        self.config = config
        self.device = torch.device(config.device)
        
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = MultimodalHateSpeechModel(config).to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
        self.model.eval()

    def evaluate_dataset(self, data_source=None, zip_path=None, split_ids_file=None, gt_json_path=None, batch_size=16, threshold=0.5, max_samples=None):
        """
        Runs evaluation over dataset source (CSV, JSON, or Zip Archive).
        Calculates Accuracy, Precision, Recall, F1-Score, and AUC/ROC.
        """
        dataset = MultimodalHateSpeechDataset(
            data_source=data_source,
            zip_path=zip_path,
            split_ids_file=split_ids_file,
            gt_json_path=gt_json_path,
            max_len=self.config.max_text_len,
            max_samples=max_samples
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_binary_probs = []
        all_binary_preds = []
        all_binary_targets = []
        all_multilabel_preds = []
        all_multilabel_targets = []
        all_gate_weights = []

        with torch.no_grad():
            for batch in dataloader:
                text_ids = batch["text_input_ids"].to(self.device)
                text_mask = batch["text_attention_mask"].to(self.device)
                pixel_vals = batch["pixel_values"].to(self.device)
                post_ids = batch["post_input_ids"].to(self.device)
                post_mask = batch["post_attention_mask"].to(self.device)
                
                outputs = self.model(
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    pixel_values=pixel_vals,
                    post_input_ids=post_ids,
                    post_attention_mask=post_mask
                )
                
                # Binary Head Predictions & Probabilities
                binary_logits = outputs["binary_logits"]
                binary_probs = torch.softmax(binary_logits, dim=-1)[:, 1].cpu().numpy() # Hate class probability
                binary_preds = (binary_probs >= threshold).astype(int)
                binary_targets = batch["binary_label"].cpu().numpy()
                
                all_binary_probs.extend(binary_probs)
                all_binary_preds.extend(binary_preds)
                all_binary_targets.extend(binary_targets)
                
                # Multi-label Head Predictions
                ml_logits = outputs["multilabel_logits"]
                ml_probs = torch.sigmoid(ml_logits).cpu().numpy()
                ml_preds = (ml_probs >= threshold).astype(int)
                ml_targets = batch["multilabel_target"].cpu().numpy()
                
                all_multilabel_preds.append(ml_preds)
                all_multilabel_targets.append(ml_targets)
                
                # Gate Weights (Visual, Text, Cross, Post)
                all_gate_weights.append(outputs["gate_weights"].cpu().numpy())

        all_binary_probs = np.array(all_binary_probs)
        all_binary_preds = np.array(all_binary_preds)
        all_binary_targets = np.array(all_binary_targets)
        
        if len(all_multilabel_preds) > 0:
            all_multilabel_preds = np.vstack(all_multilabel_preds)
            all_multilabel_targets = np.vstack(all_multilabel_targets)
            all_gate_weights = np.vstack(all_gate_weights)

        # 1. ACCURACY
        acc = accuracy_score(all_binary_targets, all_binary_preds)

        # 2. PRECISION (Macro, Micro, Hate Class)
        b_prec_macro = precision_score(all_binary_targets, all_binary_preds, average='macro', zero_division=0)
        b_prec_micro = precision_score(all_binary_targets, all_binary_preds, average='micro', zero_division=0)
        b_prec_hate = precision_score(all_binary_targets, all_binary_preds, pos_label=1, zero_division=0)

        # 3. RECALL (Macro, Micro, Hate Class)
        b_rec_macro = recall_score(all_binary_targets, all_binary_preds, average='macro', zero_division=0)
        b_rec_micro = recall_score(all_binary_targets, all_binary_preds, average='micro', zero_division=0)
        b_rec_hate = recall_score(all_binary_targets, all_binary_preds, pos_label=1, zero_division=0)

        # 4. F1 SCORE (Macro, Micro, Hate Class)
        b_f1_macro = f1_score(all_binary_targets, all_binary_preds, average='macro', zero_division=0)
        b_f1_micro = f1_score(all_binary_targets, all_binary_preds, average='micro', zero_division=0)
        b_f1_hate = f1_score(all_binary_targets, all_binary_preds, pos_label=1, zero_division=0)

        # 5. AUC / ROC
        try:
            auc_roc = roc_auc_score(all_binary_targets, all_binary_probs)
        except Exception:
            auc_roc = 0.5 # Fallback if single class in batch

        cm = confusion_matrix(all_binary_targets, all_binary_preds).tolist()

        # Category level metrics
        category_metrics = {}
        for idx, cat_name in enumerate(self.config.hate_categories):
            c_target = all_multilabel_targets[:, idx]
            c_pred = all_multilabel_preds[:, idx]
            
            c_acc = accuracy_score(c_target, c_pred)
            p = precision_score(c_target, c_pred, zero_division=0)
            r = recall_score(c_target, c_pred, zero_division=0)
            f1 = f1_score(c_target, c_pred, zero_division=0)
            
            category_metrics[cat_name] = {
                "accuracy": round(float(c_acc), 4),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f1), 4),
                "support": int(np.sum(c_target))
            }

        avg_gates = np.mean(all_gate_weights, axis=0) if len(all_gate_weights) > 0 else [0.25, 0.25, 0.25, 0.25]

        results = {
            "num_samples": len(dataset),
            "primary_metrics": {
                "accuracy": round(float(acc), 4),
                "precision": round(float(b_prec_macro), 4),
                "recall": round(float(b_rec_macro), 4),
                "f1_score": round(float(b_f1_macro), 4),
                "auc_roc": round(float(auc_roc), 4)
            },
            "binary_metrics": {
                "accuracy": round(float(acc), 4),
                "auc_roc": round(float(auc_roc), 4),
                "macro": {
                    "precision": round(float(b_prec_macro), 4),
                    "recall": round(float(b_rec_macro), 4),
                    "f1_score": round(float(b_f1_macro), 4)
                },
                "micro": {
                    "precision": round(float(b_prec_micro), 4),
                    "recall": round(float(b_rec_micro), 4),
                    "f1_score": round(float(b_f1_micro), 4)
                },
                "hate_class_specific": {
                    "precision": round(float(b_prec_hate), 4),
                    "recall": round(float(b_rec_hate), 4),
                    "f1_score": round(float(b_f1_hate), 4)
                },
                "confusion_matrix": cm
            },
            "category_metrics": category_metrics,
            "average_dynamic_gates": {
                "main_text_gate": round(float(avg_gates[0]), 4),
                "vision_gate": round(float(avg_gates[1]), 4),
                "cross_attention_gate": round(float(avg_gates[2]), 4),
                "post_text_suppression_gate": round(float(avg_gates[3]), 4)
            }
        }
        return results

    def print_summary(self, results):
        """Prints formatted summary card to console."""
        pm = results["primary_metrics"]
        bm = results["binary_metrics"]
        hate = bm["hate_class_specific"]
        gates = results["average_dynamic_gates"]

        print("=" * 75)
        print("   MULTIMODAL HATE SPEECH DETECTION EVALUATION REPORT")
        print("=" * 75)
        print(f" Total Evaluated Dataset Samples: {results['num_samples']}")
        print("-" * 75)
        print(f" 📊 PRIMARY PERFORMANCE METRICS:")
        print(f"   ► ACCURACY   : {pm['accuracy'] * 100:.2f}%")
        print(f"   ► PRECISION  : {pm['precision'] * 100:.2f}%")
        print(f"   ► RECALL     : {pm['recall'] * 100:.2f}%")
        print(f"   ► F1-SCORE   : {pm['f1_score'] * 100:.2f}%")
        print(f"   ► AUC / ROC  : {pm['auc_roc']:.4f}")
        print("-" * 75)
        print(f" HATE CLASS SPECIFIC METRICS (Positive Class = Hate Speech):")
        print(f"   ► Precision  : {hate['precision'] * 100:.2f}%")
        print(f"   ► Recall     : {hate['recall'] * 100:.2f}%")
        print(f"   ► F1-Score   : {hate['f1_score'] * 100:.2f}%")
        print("-" * 75)
        print(f" DYNAMIC CONTEXT GATING WEIGHTS:")
        print(f"   ► Main Text Gate        : {gates['main_text_gate']:.3f}")
        print(f"   ► Vision Gate           : {gates['vision_gate']:.3f}")
        print(f"   ► Cross-Attention Gate  : {gates['cross_attention_gate']:.3f}")
        print(f"   ► Post-Text Gate (Noise): {gates['post_text_suppression_gate']:.3f}")
        print("-" * 75)
        print(" PER-CATEGORY HATE SPEECH BREAKDOWN:")
        print(f" {'Category':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
        print("-" * 75)
        for cat_name, metrics in results["category_metrics"].items():
            print(f" {cat_name:<28} | {metrics['accuracy']*100:>7.2f}% | {metrics['precision']*100:>7.2f}% | {metrics['recall']*100:>7.2f}% | {metrics['f1_score']*100:>7.2f}%")
        print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Hate Speech Model on Dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset (CSV or JSON)")
    parser.add_argument("--zip_dataset", type=str, default=None, help="Path to Zip dataset archive (e.g. Multimodal Hate Speech Archive.zip)")
    parser.add_argument("--split_ids", type=str, default="splits/test_ids.txt", help="Path to split IDs file")
    parser.add_argument("--gt_json", type=str, default="MMHS150K_GT.json", help="Path to GT json file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_json", type=str, default="evaluation_results.json", help="Path to save results JSON")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of dataset samples to evaluate")
    args = parser.parse_args()

    evaluator = Evaluator(checkpoint_path=args.checkpoint)
    
    if args.zip_dataset or os.path.exists("Multimodal Hate Speech Archive.zip"):
        zip_path = args.zip_dataset if args.zip_dataset else "Multimodal Hate Speech Archive.zip"
        print(f"Evaluating Zip Dataset Archive: {zip_path} (Split: {args.split_ids})")
        results = evaluator.evaluate_dataset(
            zip_path=zip_path,
            split_ids_file=args.split_ids,
            gt_json_path=args.gt_json,
            max_samples=args.max_samples
        )
    else:
        print(f"Evaluating Dataset File: {args.dataset}")
        results = evaluator.evaluate_dataset(data_source=args.dataset, max_samples=args.max_samples)

    evaluator.print_summary(results)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved complete evaluation metrics report to: {args.output_json}")

if __name__ == "__main__":
    main()
