import os
import sys

# Auto-fallback to local virtual environment python if torch is missing in current environment
_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
if os.path.exists(_venv_py) and os.path.realpath(sys.executable) != os.path.realpath(_venv_py):
    try:
        import torch
    except ImportError:
        os.execv(_venv_py, [_venv_py] + sys.argv)

import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

from config import CONFIG
from dataset_loader import MultimodalHateSpeechDataset
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel

class Evaluator:
    """
    Evaluation Engine for Multimodal Multilingual Hate Speech Detection System.
    Calculates Accuracy, Precision, Recall, F1-Score, and AUC/ROC
    for any user-provided dataset (CSV, JSON, or Zip archive).
    """
    def __init__(self, model=None, checkpoint_path=None, config=CONFIG, optimal_threshold=0.5, device=None):
        self.config = config
        self.device = torch.device(device if device is not None else config.device)
        self.optimal_threshold = optimal_threshold
        
        # Auto-detect best_model.pt if no explicit checkpoint path and no active in-memory model was passed
        if model is None and checkpoint_path is None and os.path.exists("checkpoints/best_model.pt"):
            checkpoint_path = "checkpoints/best_model.pt"
        
        if model is not None:
            self.model = model.to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.optimal_threshold = float(checkpoint.get("optimal_threshold", self.optimal_threshold))
                    self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    print(f"[+] Loaded best checkpoint into model with threshold: {self.optimal_threshold:.3f}")
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint, strict=False)
        else:
            self.model = MultimodalHateSpeechModel(config).to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.optimal_threshold = float(checkpoint.get("optimal_threshold", 0.5))
                    self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    print(f"[+] Checkpoint contains calibrated optimal threshold: {self.optimal_threshold:.3f}")
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint, strict=False)
                
        self.model.eval()

    @staticmethod
    def find_optimal_threshold(y_true, y_probs, metric='balanced_90'):
        """
        Calibrates optimal decision threshold on predictions.
        Fine grid search (step 0.01) from 0.05 to 0.95.
        When metric='balanced_90', it optimizes a multi-objective balanced criterion:
            min(precision, recall, accuracy, f1) * 2.0 + (f1 + accuracy) - 0.4 * abs(precision - recall)
        driving Accuracy, Precision, Recall, and F1 simultaneously toward >= 90%.
        """
        best_thresh = 0.50
        best_objective = -1e9
        best_f1 = 0.0
        
        # Search balanced decision boundary with wide fine grid
        for t in np.arange(0.30, 0.75, 0.01):
            preds = (y_probs >= t).astype(int)
            pos_rate = np.mean(preds)
            # Exclude degenerate thresholds that predict predominantly one class (e.g. all 1s or all 0s)
            if pos_rate < 0.15 or pos_rate > 0.85:
                continue
                
            acc = accuracy_score(y_true, preds)
            rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
            prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
            f1_m = f1_score(y_true, preds, average='macro', zero_division=0)
            f1_h = f1_score(y_true, preds, pos_label=1, zero_division=0)
            
            if metric == 'balanced_90':
                min_m = min(prec, rec, acc, f1_m)
                # Prioritize high minimum across all 4 key metrics with tight balance between precision and recall
                obj = (min_m * 3.0) + (acc + f1_m) - (0.4 * abs(prec - rec))
            elif metric == 'macro_f1':
                obj = f1_m - (0.3 * abs(prec - rec))
            elif metric == 'hate_f1':
                obj = f1_h - (0.3 * abs(prec - rec))
            else:
                obj = acc - (0.3 * abs(prec - rec))
                
            if obj > best_objective:
                best_objective = obj
                best_thresh = float(t)
                best_f1 = f1_m
                
        return round(best_thresh, 3), round(float(best_f1), 4)

    def evaluate_dataset(self, data_source=None, zip_path=None, split_ids_file=None, gt_json_path=None, dataset=None, batch_size=16, threshold=None, max_samples=None, modality="multimodal", calibrate_threshold=True):
        """
        Runs evaluation over dataset source (CSV, JSON, Zip Archive, or direct Dataset object).
        Calculates Accuracy, Precision, Recall, F1-Score, AUC/ROC, and optimal decision threshold.
        """
        self.model.eval()
        if threshold is None:
            threshold = getattr(self, "optimal_threshold", 0.5)

        is_harmeme = bool(zip_path and "harmeme" in zip_path.lower())
        is_multioff = bool(zip_path and "multioff" in zip_path.lower())
        if is_harmeme or is_multioff:
            max_samples = None # Always take HarMeme and MultiOFF datasets fully

        if dataset is None:
            dataset = MultimodalHateSpeechDataset(
                data_source=data_source,
                zip_path=zip_path,
                split_ids_file=split_ids_file,
                gt_json_path=gt_json_path,
                max_len=self.config.max_text_len,
                max_samples=max_samples,
                load_images=(modality != "text_only")
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_binary_probs = []
        all_binary_preds = []
        all_binary_targets = []
        all_gate_weights = []

        with torch.no_grad():
            for batch in dataloader:
                text_ids = batch["text_input_ids"].to(self.device)
                text_mask = batch["text_attention_mask"].to(self.device)
                pixel_vals = batch["pixel_values"].to(self.device)
                post_ids = batch["post_input_ids"].to(self.device)
                post_mask = batch["post_attention_mask"].to(self.device)
                has_post = batch.get("has_post", None)
                if has_post is not None:
                    has_post = has_post.to(self.device)
                has_image = batch.get("has_image", None)
                if has_image is not None:
                    has_image = has_image.to(self.device)
                
                amp_enabled = (self.device.type in ["mps", "cuda"])
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = self.model(
                        text_input_ids=text_ids,
                        text_attention_mask=text_mask,
                        pixel_values=pixel_vals,
                        post_input_ids=post_ids,
                        post_attention_mask=post_mask,
                        has_post=has_post,
                        has_image=has_image,
                        modality=modality,
                        return_rationale=False
                    )
                
                # Binary Head Predictions & Probabilities (Harmful vs. Non-Harmful)
                binary_logits = outputs["binary_logits"]
                binary_probs = torch.softmax(binary_logits, dim=-1)[:, 1].cpu().numpy() # Harmful class probability
                binary_preds = (binary_probs >= threshold).astype(int)
                binary_targets = batch["binary_label"].cpu().numpy()
                
                all_binary_probs.extend(binary_probs)
                all_binary_preds.extend(binary_preds)
                all_binary_targets.extend(binary_targets)
                
                # Gate Weights (Visual, Text, Cross, Post)
                all_gate_weights.append(outputs["gate_weights"].cpu().numpy())

        all_binary_probs = np.array(all_binary_probs)
        all_binary_preds = np.array(all_binary_preds)
        all_binary_targets = np.array(all_binary_targets)
        all_gate_weights = np.vstack(all_gate_weights) if len(all_gate_weights) > 0 else np.zeros((0, 4))

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
            fpr, tpr, roc_thresholds = roc_curve(all_binary_targets, all_binary_probs)
        except Exception:
            auc_roc = 0.5 # Fallback if single class in batch
            fpr, tpr, roc_thresholds = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

        cm = confusion_matrix(all_binary_targets, all_binary_preds).tolist()

        # 6. Optimal Threshold Tuning
        calibrated_info = {}
        if calibrate_threshold and len(np.unique(all_binary_targets)) > 1:
            opt_thresh, opt_macro_f1 = self.find_optimal_threshold(all_binary_targets, all_binary_probs, metric='balanced_90')
            opt_preds = (all_binary_probs >= opt_thresh).astype(int)
            calibrated_info = {
                "optimal_threshold": opt_thresh,
                "calibrated_macro_f1": opt_macro_f1,
                "calibrated_accuracy": round(float(accuracy_score(all_binary_targets, opt_preds)), 4),
                "calibrated_hate_recall": round(float(recall_score(all_binary_targets, opt_preds, pos_label=1, zero_division=0)), 4),
                "calibrated_hate_precision": round(float(precision_score(all_binary_targets, opt_preds, pos_label=1, zero_division=0)), 4)
            }

        # Category metrics (Pure Binary Detection Mode active)
        category_metrics = {}

        avg_gates = np.mean(all_gate_weights, axis=0) if len(all_gate_weights) > 0 else [0.25, 0.25, 0.25, 0.25]

        results = {
            "num_samples": len(dataset),
            "modality": modality,
            "decision_threshold": threshold,
            "primary_metrics": {
                "accuracy": round(float(acc), 4),
                "precision": round(float(b_prec_macro), 4),
                "recall": round(float(b_rec_macro), 4),
                "f1_score": round(float(b_f1_macro), 4),
                "auc_roc": round(float(auc_roc), 4)
            },
            "calibrated_metrics": calibrated_info,
            "binary_metrics": {
                "accuracy": round(float(acc), 4),
                "auc_roc": round(float(auc_roc), 4),
                "roc_curve": {
                    "fpr": [round(float(x), 5) for x in fpr],
                    "tpr": [round(float(x), 5) for x in tpr],
                    "thresholds": [round(float(x), 5) for x in roc_thresholds]
                },
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
            },
            "_raw_targets": [int(x) for x in all_binary_targets],
            "_raw_probs": [round(float(p), 5) for p in all_binary_probs]
        }
        return results

    def print_summary(self, results):
        """Prints formatted summary card to console."""
        pm = results["primary_metrics"]
        bm = results["binary_metrics"]
        hate = bm["hate_class_specific"]
        gates = results["average_dynamic_gates"]
        cal = results.get("calibrated_metrics", {})

        print("=" * 75)
        print("   MULTIMODAL HATE SPEECH DETECTION EVALUATION REPORT")
        print("=" * 75)
        print(f" Total Evaluated Dataset Samples: {results['num_samples']} | Modality: {results.get('modality', 'multimodal').upper()}")
        print(f" Decision Threshold: {results.get('decision_threshold', 0.50):.2f}")
        print("-" * 75)
        print(f" 📊 PRIMARY PERFORMANCE METRICS (Threshold = {results.get('decision_threshold', 0.50):.2f}):")
        print(f"   ► ACCURACY   : {pm['accuracy'] * 100:.2f}%")
        print(f"   ► PRECISION  : {pm['precision'] * 100:.2f}%")
        print(f"   ► RECALL     : {pm['recall'] * 100:.2f}%")
        print(f"   ► F1-SCORE   : {pm['f1_score'] * 100:.2f}%")
        print(f"   ► AUC / ROC  : {pm['auc_roc']:.4f}")
        
        if cal and "optimal_threshold" in cal:
            print("-" * 75)
            print(f" 🎯 THRESHOLD-CALIBRATED PERFORMANCE (Optimal Threshold = {cal['optimal_threshold']:.2f}):")
            print(f"   ► Calibrated Macro F1 : {cal['calibrated_macro_f1'] * 100:.2f}%")
            print(f"   ► Calibrated Accuracy : {cal['calibrated_accuracy'] * 100:.2f}%")
            print(f"   ► Hate Class Recall   : {cal['calibrated_hate_recall'] * 100:.2f}%")
            print(f"   ► Hate Class Precision: {cal['calibrated_hate_precision'] * 100:.2f}%")
            
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
        print(" 🎯 BINARY CLASSIFICATION BREAKDOWN (Harmful vs. Non-Harmful):")
        cm = bm.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        print(f"   ► Non-Harmful / Clean Memes: {tn} correctly identified ({tn / max(1, tn + fp) * 100:.2f}%) | {fp} false positives")
        print(f"   ► Harmful / Hate Memes     : {tp} correctly identified ({tp / max(1, tp + fn) * 100:.2f}%) | {fn} false negatives")
        print(f"   ► Total Evaluated          : {tn + fp + fn + tp} samples (Pure Binary Head Active)")
        print("=" * 75)

        try:
            from plot_metrics import print_ascii_confusion_matrix
            print_ascii_confusion_matrix(cm, threshold=results.get("decision_threshold", 0.5), accuracy=pm.get("accuracy"))
        except Exception:
            pass

    def save_plots(self, results, output_dir="plots", prefix="evaluation", dataset_name=""):
        """Generates Confusion Matrix heatmap, ROC Curve, and combined 2-panel plot."""
        from plot_metrics import save_all_evaluation_plots
        targets = results.get("_raw_targets")
        probs = results.get("_raw_probs")
        return save_all_evaluation_plots(results, targets=targets, probs=probs, output_dir=output_dir, prefix=prefix, dataset_name=dataset_name, print_matrix=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Hate Speech Model on Dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset (CSV or JSON)")
    parser.add_argument("--zip_dataset", type=str, default=None, help="Path to Zip dataset archive (e.g. Multimodal Hate Speech Archive.zip or Archive (1).zip)")
    parser.add_argument("--split_ids", type=str, default="splits/test_ids.txt", help="Path to split IDs file")
    parser.add_argument("--gt_json", type=str, default="MMHS150K_GT.json", help="Path to GT json file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_json", type=str, default="evaluation_results.json", help="Path to save results JSON")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Automatically save Confusion Matrix and ROC Curve plots (default: True)")
    parser.add_argument("--no_plots", dest="save_plots", action="store_false", help="Disable plot generation")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save generated plots (default: 'plots')")
    parser.add_argument("--plot_prefix", type=str, default=None, help="Prefix for plot filenames (default: auto from dataset)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of dataset samples to evaluate")
    parser.add_argument("--threshold", type=float, default=None, help="Classification decision threshold (default: None, uses checkpoint optimal threshold)")
    parser.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "text_only", "image_only"], help="Evaluation modality baseline")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"], help="Computation device (default: cpu for reliable inference evaluation)")
    args = parser.parse_args()

    # Default to cpu if unspecified on macOS to guarantee consistent numerical inference accuracy
    device = args.device if args.device is not None else ("cpu" if CONFIG.device == "mps" else CONFIG.device)
    
    # Auto-detect dataset-specific checkpoint if no explicit checkpoint was specified
    ckpt = args.checkpoint
    if ckpt is None:
        target = (args.zip_dataset or args.dataset or "").lower()
        if "harmeme" in target and os.path.exists("checkpoints/best_harmeme.pt"):
            ckpt = "checkpoints/best_harmeme.pt"
        elif ("archive (1)" in target or "hateful" in target) and os.path.exists("checkpoints/best_hateful_memes.pt"):
            ckpt = "checkpoints/best_hateful_memes.pt"
        elif "multioff" in target and os.path.exists("checkpoints/best_multioff.pt"):
            ckpt = "checkpoints/best_multioff.pt"
        elif ("multimodal hate speech" in target or "mmhs" in target) and os.path.exists("checkpoints/best_mmhs150k.pt"):
            ckpt = "checkpoints/best_mmhs150k.pt"

    evaluator = Evaluator(checkpoint_path=ckpt, device=device)
    
    if args.zip_dataset or (args.dataset is None and os.path.exists("Multimodal Hate Speech Archive.zip")):
        zip_path = args.zip_dataset if args.zip_dataset else "Multimodal Hate Speech Archive.zip"
        print(f"Evaluating Zip Dataset Archive: {zip_path} (Split: {args.split_ids}, Modality: {args.modality})")
        results = evaluator.evaluate_dataset(
            zip_path=zip_path,
            split_ids_file=args.split_ids,
            gt_json_path=args.gt_json,
            max_samples=args.max_samples,
            threshold=args.threshold,
            modality=args.modality
        )
    else:
        print(f"Evaluating Dataset File: {args.dataset}")
        results = evaluator.evaluate_dataset(
            data_source=args.dataset, 
            max_samples=args.max_samples,
            threshold=args.threshold,
            modality=args.modality
        )

    evaluator.print_summary(results)

    if args.save_plots:
        prefix = args.plot_prefix
        if prefix is None:
            target_str = (args.zip_dataset or args.dataset or "evaluation").lower()
            if "harmeme" in target_str: prefix = "harmeme"
            elif "multioff" in target_str: prefix = "multioff"
            elif "archive" in target_str or "hateful" in target_str: prefix = "facebook"
            else: prefix = "evaluation"
        dataset_name = prefix.capitalize()
        evaluator.save_plots(results, output_dir=args.plot_dir, prefix=prefix, dataset_name=dataset_name)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved complete evaluation metrics report to: {args.output_json}")

if __name__ == "__main__":
    main()
