import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from config import CONFIG
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel
from text_deobfuscator import TextDeobfuscator

class MultimodalHateSpeechTerminalEvaluator:
    """
    Terminal Evaluator for Multimodal Multilingual Hate Speech Detection System.
    Outputs Accuracy, Precision, Recall, F1-Score, AUC/ROC, Confusion Matrix,
    and Confidence Score Matrix directly in the terminal output.
    """
    def __init__(self, checkpoint_path=None, config=CONFIG):
        self.config = config
        self.device = torch.device(config.device)
        print(f"\n[+] Initializing Multimodal Multilingual Model on {self.device}...")
        self.model = MultimodalHateSpeechModel(config).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[+] Loading trained model checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()

    def run_evaluation(self, zip_path=None, split_ids_file="splits/test_ids.txt", gt_json_path="MMHS150K_GT.json", dataset_path=None, max_samples=1000, batch_size=16, threshold=0.5):
        """
        Executes evaluation and prints detailed metrics report to terminal.
        """
        # Determine dataset source
        if zip_path and os.path.exists(zip_path):
            print(f"[+] Loading dataset from Zip Archive: {zip_path}")
            dataset = MultimodalHateSpeechDataset(
                data_source=None,
                zip_path=zip_path,
                split_ids_file=split_ids_file,
                gt_json_path=gt_json_path,
                max_len=self.config.max_text_len,
                max_samples=max_samples
            )
        elif dataset_path and os.path.exists(dataset_path):
            print(f"[+] Loading dataset file: {dataset_path}")
            dataset = MultimodalHateSpeechDataset(
                data_source=dataset_path,
                max_len=self.config.max_text_len,
                max_samples=max_samples
            )
        else:
            default_csv = "sample_data/multilingual_hate_speech_benchmark.csv"
            print(f"[+] Loading benchmark dataset: {default_csv}")
            dataset = MultimodalHateSpeechDataset(
                data_source=default_csv,
                max_len=self.config.max_text_len,
                max_samples=max_samples
            )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"[+] Running evaluation inference on {len(dataset)} samples...")

        all_probs = []
        all_preds = []
        all_targets = []
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
                
                # Binary Softmax Probabilities
                logits = outputs["binary_logits"]
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy() # Probability of Hate
                preds = (probs >= threshold).astype(int)
                targets = batch["binary_label"].cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)
                
                # Multi-label prediction
                ml_logits = outputs["multilabel_logits"]
                ml_probs = torch.sigmoid(ml_logits).cpu().numpy()
                ml_preds = (ml_probs >= threshold).astype(int)
                ml_targets = batch["multilabel_target"].cpu().numpy()
                
                all_multilabel_preds.append(ml_preds)
                all_multilabel_targets.append(ml_targets)
                
                # Dynamic Gate Weights
                all_gate_weights.append(outputs["gate_weights"].cpu().numpy())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        if len(all_multilabel_preds) > 0:
            all_multilabel_preds = np.vstack(all_multilabel_preds)
            all_multilabel_targets = np.vstack(all_multilabel_targets)
            all_gate_weights = np.vstack(all_gate_weights)

        # 1. ACCURACY
        acc = accuracy_score(all_targets, all_preds)

        # 2. PRECISION
        prec_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        prec_micro = precision_score(all_targets, all_preds, average='micro', zero_division=0)
        prec_hate = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        prec_normal = precision_score(all_targets, all_preds, pos_label=0, zero_division=0)

        # 3. RECALL
        rec_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        rec_micro = recall_score(all_targets, all_preds, average='micro', zero_division=0)
        rec_hate = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        rec_normal = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)

        # 4. F1 SCORE
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
        f1_hate = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
        f1_normal = f1_score(all_targets, all_preds, pos_label=0, zero_division=0)

        # 5. AUC / ROC
        try:
            auc_roc = roc_auc_score(all_targets, all_probs)
        except Exception:
            auc_roc = 0.5

        # 6. CONFUSION MATRIX
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # 7. CONFIDENCE SCORE MATRIX & STATS
        hate_mask = (all_targets == 1)
        norm_mask = (all_targets == 0)
        
        avg_conf_hate_true = float(np.mean(all_probs[hate_mask])) if np.sum(hate_mask) > 0 else 0.0
        avg_conf_norm_true = float(np.mean(1.0 - all_probs[norm_mask])) if np.sum(norm_mask) > 0 else 0.0
        overall_mean_confidence = float(np.mean(np.maximum(all_probs, 1.0 - all_probs)))

        # Category Breakdown
        category_metrics = {}
        for idx, cat_name in enumerate(self.config.hate_categories):
            c_target = all_multilabel_targets[:, idx]
            c_pred = all_multilabel_preds[:, idx]
            
            c_acc = accuracy_score(c_target, c_pred)
            p = precision_score(c_target, c_pred, zero_division=0)
            r = recall_score(c_target, c_pred, zero_division=0)
            f1 = f1_score(c_target, c_pred, zero_division=0)
            
            category_metrics[cat_name] = {
                "accuracy": c_acc,
                "precision": p,
                "recall": r,
                "f1_score": f1,
                "support": int(np.sum(c_target))
            }

        avg_gates = np.mean(all_gate_weights, axis=0) if len(all_gate_weights) > 0 else [0.25, 0.25, 0.25, 0.25]

        # PRINT TERMINAL REPORT
        self._print_terminal_report(
            num_samples=len(dataset),
            acc=acc,
            prec_macro=prec_macro, rec_macro=rec_macro, f1_macro=f1_macro,
            prec_hate=prec_hate, rec_hate=rec_hate, f1_hate=f1_hate,
            prec_normal=prec_normal, rec_normal=rec_normal, f1_normal=f1_normal,
            auc_roc=auc_roc,
            tn=tn, fp=fp, fn=fn, tp=tp,
            avg_conf_hate=avg_conf_hate_true, avg_conf_norm=avg_conf_norm_true, overall_mean_conf=overall_mean_confidence,
            avg_gates=avg_gates,
            category_metrics=category_metrics
        )

    def _print_terminal_report(self, num_samples, acc, prec_macro, rec_macro, f1_macro, prec_hate, rec_hate, f1_hate, prec_normal, rec_normal, f1_normal, auc_roc, tn, fp, fn, tp, avg_conf_hate, avg_conf_norm, overall_mean_conf, avg_gates, category_metrics):
        total = tn + fp + fn + tp
        
        print("\n" + "=" * 80)
        print("  🎯 MULTIMODAL HATE SPEECH DETECTION MODEL PERFORMANCE REPORT")
        print("=" * 80)
        print(f" Total Evaluated Dataset Samples : {num_samples}")
        print(f" Classification Threshold         : 0.50")
        print("=" * 80)
        
        print("\n 📌 1. PRIMARY METRICS SUMMARY:")
        print(" ┌──────────────────────┬─────────────────┬─────────────────┐")
        print(f" │ Metric               │ Percentage (%)  │ Decimal Score   │")
        print(" ├──────────────────────┼─────────────────┼─────────────────┤")
        print(f" │ ACCURACY             │ {acc * 100:>13.2f}% │ {acc:>15.4f} │")
        print(f" │ PRECISION (Macro)    │ {prec_macro * 100:>13.2f}% │ {prec_macro:>15.4f} │")
        print(f" │ RECALL (Macro)       │ {rec_macro * 100:>13.2f}% │ {rec_macro:>15.4f} │")
        print(f" │ F1-SCORE (Macro)     │ {f1_macro * 100:>13.2f}% │ {f1_macro:>15.4f} │")
        print(f" │ AUC / ROC            │ {'N/A':>14}  │ {auc_roc:>15.4f} │")
        print(" └──────────────────────┴─────────────────┴─────────────────┘")

        print("\n 📊 2. CLASS-SPECIFIC PERFORMANCE (Hate vs. Normal):")
        print(" ┌──────────────────────┬──────────────┬──────────────┬──────────────┐")
        print(" │ Class                │ Precision    │ Recall       │ F1-Score     │")
        print(" ├──────────────────────┼──────────────┼──────────────┼──────────────┤")
        print(f" │ Hate Speech (1)      │ {prec_hate * 100:>10.2f}% │ {rec_hate * 100:>10.2f}% │ {f1_hate * 100:>10.2f}% │")
        print(f" │ Normal / Clean (0)   │ {prec_normal * 100:>10.2f}% │ {rec_normal * 100:>10.2f}% │ {f1_normal * 100:>10.2f}% │")
        print(" └──────────────────────┴──────────────┴──────────────┴──────────────┘")

        print("\n 🧩 3. CONFUSION MATRIX:")
        print(" ┌──────────────────────────────────┬──────────────────────────────────┐")
        print(" │                                  │          PREDICTED CLASS         │")
        print(" │ ACTUAL CLASS                     ├─────────────────┬────────────────┤")
        print(" │                                  │ Normal (0)      │ Hate Speech (1)│")
        print(" ├──────────────────────────────────┼─────────────────┼────────────────┤")
        print(f" │ Actual Normal (0)                │ TN = {tn:<10} │ FP = {fp:<10} │")
        print(f" │ Actual Hate Speech (1)           │ FN = {fn:<10} │ TP = {tp:<10} │")
        print(" └──────────────────────────────────┴─────────────────┴────────────────┘")
        if total > 0:
            print(f"   • True Positive Rate (Sensitivity / Recall)  : {tp / max(1, tp + fn) * 100:.2f}%")
            print(f"   • True Negative Rate (Specificity)           : {tn / max(1, tn + fp) * 100:.2f}%")
            print(f"   • False Positive Rate (Type I Error Rate)    : {fp / max(1, fp + tn) * 100:.2f}%")
            print(f"   • False Negative Rate (Type II Error Rate)   : {fn / max(1, fn + tp) * 100:.2f}%")

        print("\n 📈 4. CONFIDENCE SCORE & PROBABILITY MATRIX:")
        print(" ┌─────────────────────────────────────────────────┬───────────────────┐")
        print(" │ Confidence Metric                               │ Score             │")
        print(" ├─────────────────────────────────────────────────┼───────────────────┤")
        print(f" │ Avg Model Confidence on Hate Samples           │ {avg_conf_hate * 100:>15.2f}%  │")
        print(f" │ Avg Model Confidence on Normal Samples         │ {avg_conf_norm * 100:>15.2f}%  │")
        print(f" │ Overall Mean Prediction Certainty Score          │ {overall_mean_conf * 100:>15.2f}%  │")
        print(" └─────────────────────────────────────────────────┴───────────────────┘")

        print("\n ⚡ 5. DYNAMIC CONTEXT GATING WEIGHTS:")
        print(f"   • Main Text Stream Gate (Caption + OCR) : {avg_gates[0]:.4f}")
        print(f"   • ViT Visual Stream Gate (Image)        : {avg_gates[1]:.4f}")
        print(f"   • DDCAF Cross-Attention Gate            : {avg_gates[2]:.4f}")
        print(f"   • Post Noise Suppression Gate           : {avg_gates[3]:.4f}")

        print("\n 🏷️  6. PER-CATEGORY HATE SPEECH BREAKDOWN:")
        print(" ┌──────────────────────────────┬───────────┬───────────┬───────────┬───────────┬─────────┐")
        print(" │ Category                     │ Accuracy  │ Precision │ Recall    │ F1-Score  │ Support │")
        print(" ├──────────────────────────────┼───────────┼───────────┼───────────┼───────────┼─────────┤")
        for cat_name, metrics in category_metrics.items():
            print(f" │ {cat_name:<28} │ {metrics['accuracy']*100:>8.2f}% │ {metrics['precision']*100:>8.2f}% │ {metrics['recall']*100:>8.2f}% │ {metrics['f1_score']*100:>8.2f}% │ {metrics['support']:<7} │")
        print(" └──────────────────────────────┴───────────┴───────────┴───────────┴───────────┴─────────┘")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Terminal Evaluator for Multimodal Hate Speech System")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV/JSON dataset file")
    parser.add_argument("--zip_dataset", type=str, default="Multimodal Hate Speech Archive.zip", help="Path to Zip dataset archive")
    parser.add_argument("--split_ids", type=str, default="splits/test_ids.txt", help="Path to split IDs file")
    parser.add_argument("--gt_json", type=str, default="MMHS150K_GT.json", help="Path to GT json file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of samples to evaluate")
    args = parser.parse_args()

    evaluator = MultimodalHateSpeechTerminalEvaluator(checkpoint_path=args.checkpoint)
    
    if args.dataset and os.path.exists(args.dataset):
        evaluator.run_evaluation(dataset_path=args.dataset, max_samples=args.max_samples)
    elif os.path.exists(args.zip_dataset):
        evaluator.run_evaluation(
            zip_path=args.zip_dataset,
            split_ids_file=args.split_ids,
            gt_json_path=args.gt_json,
            max_samples=args.max_samples
        )
    else:
        evaluator.run_evaluation(max_samples=args.max_samples)

if __name__ == "__main__":
    main()
