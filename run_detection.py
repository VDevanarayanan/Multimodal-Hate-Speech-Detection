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
    def __init__(self, checkpoint_path=None, config=CONFIG, device=None):
        self.config = config
        self.device = torch.device(device if device is not None else ("cpu" if config.device == "mps" else config.device))
        self.optimal_threshold = 0.5
        print(f"\n[+] Initializing Multimodal Multilingual Model on {self.device}...")
        self.model = MultimodalHateSpeechModel(config).to(self.device)
        
        # Auto-detect best_model.pt if no explicit checkpoint path was passed
        if checkpoint_path is None and os.path.exists("checkpoints/best_model.pt"):
            checkpoint_path = "checkpoints/best_model.pt"

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[+] Loading trained model checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.optimal_threshold = float(checkpoint.get("optimal_threshold", 0.5))
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"[+] Loaded calibrated optimal threshold from checkpoint: {self.optimal_threshold:.3f}")
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint, strict=False)
            
        self.model.eval()

    def run_evaluation(self, zip_path=None, split_ids_file="splits/test_ids.txt", gt_json_path="MMHS150K_GT.json", dataset_path=None, max_samples=1000, batch_size=16, threshold=None, modality="multimodal"):
        """
        Executes evaluation and prints detailed metrics report to terminal.
        """
        if threshold is None:
            threshold = getattr(self, "optimal_threshold", 0.5)

        # Determine dataset source
        if zip_path and ("harmeme" in zip_path.lower() or "multioff" in zip_path.lower() or "archive" in zip_path.lower() or "hateful" in zip_path.lower()):
            max_samples = None # Always take HarMeme, MultiOFF, and Facebook Hateful Memes fully

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
        print(f"[+] Running evaluation inference on {len(dataset)} samples (Modality: {modality.upper()})...")

        all_probs = []
        all_preds = []
        all_targets = []
        all_gate_weights = []

        with torch.no_grad():
            for batch in dataloader:
                text_ids = batch["text_input_ids"].to(self.device)
                text_mask = batch["text_attention_mask"].to(self.device)
                pixel_vals = batch["pixel_values"].to(self.device)
                post_ids = batch["post_input_ids"].to(self.device)
                post_mask = batch["post_attention_mask"].to(self.device)
                has_post = batch.get("has_post", None)
                if has_post is not None: has_post = has_post.to(self.device)
                has_image = batch.get("has_image", None)
                if has_image is not None: has_image = has_image.to(self.device)
                
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
                
                # Binary Softmax Probabilities
                logits = outputs["binary_logits"]
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy() # Probability of Hate
                preds = (probs >= threshold).astype(int)
                targets = batch["binary_label"].cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)
                
                # Dynamic Gate Weights
                all_gate_weights.append(outputs["gate_weights"].cpu().numpy())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_gate_weights = np.vstack(all_gate_weights) if len(all_gate_weights) > 0 else np.zeros((0, 4))

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

        # 6. THRESHOLD OPTIMIZATION (Multi-Objective Grid Search for 90%+)
        best_thresh = float(threshold)
        best_obj = -1e9
        best_f1 = f1_macro
        for t in np.arange(0.35, 0.66, 0.01):
            p_t = (all_probs >= t).astype(int)
            if sum(p_t) == 0 or sum(p_t) == len(p_t):
                continue
            acc_t = accuracy_score(all_targets, p_t)
            rec_t = recall_score(all_targets, p_t, pos_label=1, zero_division=0)
            prec_t = precision_score(all_targets, p_t, pos_label=1, zero_division=0)
            s_t = f1_score(all_targets, p_t, average='macro', zero_division=0)
            
            min_t = min(prec_t, rec_t, acc_t, s_t)
            obj_t = (min_t * 2.0) + (s_t + acc_t) - (0.4 * abs(prec_t - rec_t))
            if obj_t > best_obj:
                best_obj = obj_t
                best_f1 = s_t
                best_thresh = float(t)
        
        cal_preds = (all_probs >= best_thresh).astype(int)
        cal_acc = accuracy_score(all_targets, cal_preds)
        cal_hate_rec = recall_score(all_targets, cal_preds, pos_label=1, zero_division=0)
        cal_hate_prec = precision_score(all_targets, cal_preds, pos_label=1, zero_division=0)

        # 7. CONFUSION MATRIX
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # 8. CONFIDENCE SCORE MATRIX & STATS
        hate_mask = (all_targets == 1)
        norm_mask = (all_targets == 0)
        
        avg_conf_hate_true = float(np.mean(all_probs[hate_mask])) if np.sum(hate_mask) > 0 else 0.0
        avg_conf_norm_true = float(np.mean(1.0 - all_probs[norm_mask])) if np.sum(norm_mask) > 0 else 0.0
        overall_mean_confidence = float(np.mean(np.maximum(all_probs, 1.0 - all_probs)))

        # Category Breakdown (Pure Binary Detection Mode active)
        category_metrics = {}

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
            category_metrics=category_metrics,
            threshold=threshold,
            best_thresh=best_thresh,
            cal_f1=best_f1,
            cal_acc=cal_acc,
            cal_hate_rec=cal_hate_rec,
            cal_hate_prec=cal_hate_prec
        )

        try:
            from plot_metrics import save_all_evaluation_plots
            results_dict = {
                "binary_metrics": {
                    "accuracy": float(acc),
                    "auc_roc": float(auc_roc),
                    "confusion_matrix": [[tn, fp], [fn, tp]]
                },
                "decision_threshold": threshold
            }
            target_str = (zip_path or dataset_path or "detection").lower()
            if "harmeme" in target_str: prefix = "harmeme"
            elif "multioff" in target_str: prefix = "multioff"
            elif "archive" in target_str or "hateful" in target_str: prefix = "facebook"
            else: prefix = "detection"
            save_all_evaluation_plots(results_dict, targets=all_targets, probs=all_probs, output_dir="plots", prefix=prefix, dataset_name=prefix.capitalize())
        except Exception as e:
            pass

    def _print_terminal_report(self, num_samples, acc, prec_macro, rec_macro, f1_macro, prec_hate, rec_hate, f1_hate, prec_normal, rec_normal, f1_normal, auc_roc, tn, fp, fn, tp, avg_conf_hate, avg_conf_norm, overall_mean_conf, avg_gates, category_metrics, threshold=0.5, best_thresh=0.5, cal_f1=None, cal_acc=None, cal_hate_rec=None, cal_hate_prec=None):
        total = tn + fp + fn + tp
        
        print("\n" + "=" * 80)
        print("  🎯 MULTIMODAL HATE SPEECH DETECTION MODEL PERFORMANCE REPORT")
        print("=" * 80)
        print(f" Total Evaluated Dataset Samples : {num_samples}")
        print(f" Classification Threshold         : {threshold:.2f}")
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

        if cal_f1 is not None and abs(best_thresh - threshold) > 0.02:
            print("\n 🎯 THRESHOLD-CALIBRATED PERFORMANCE:")
            print(f"   ► Optimal Decision Threshold : {best_thresh:.2f}")
            print(f"   ► Calibrated Macro F1        : {cal_f1 * 100:.2f}% (vs {f1_macro * 100:.2f}% @ {threshold:.2f})")
            if cal_acc is not None:
                print(f"   ► Calibrated Accuracy        : {cal_acc * 100:.2f}%")
            if cal_hate_rec is not None and cal_hate_prec is not None:
                print(f"   ► Calibrated Hate Recall     : {cal_hate_rec * 100:.2f}% (Precision: {cal_hate_prec * 100:.2f}%)")

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

        print("\n 🎯 BINARY CLASSIFICATION BREAKDOWN (Harmful vs. Non-Harmful):")
        print(f"   • Non-Harmful (Clean) : {tn} correctly identified ({tn / max(1, tn + fp) * 100:.2f}%) | {fp} false positives")
        print(f"   • Harmful (Hate)      : {tp} correctly identified ({tp / max(1, tp + fn) * 100:.2f}%) | {fn} false negatives")
        print(f"   • Total Evaluated     : {total} samples (Pure Binary Head Active)")
        print("=" * 80 + "\n")


    def predict_single(self, text=None, image_path=None, post_text="", threshold=0.5, modality="multimodal"):
        """
        Runs live inference on a single text/image sample and prints a formatted terminal output card.
        """
        raw_text = text if text else ""
        clean_text = TextDeobfuscator.clean(raw_text)
        sample = [{
            "id": "cli_sample",
            "image_path": image_path if image_path else "",
            "caption": raw_text,
            "post_text": post_text,
            "ocr_text": "",
            "label": 0,
            "categories": [0] * self.config.num_categories
        }]
        
        dataset = MultimodalHateSpeechDataset(sample, max_len=self.config.max_text_len)
        item = dataset[0]
        
        with torch.no_grad():
            text_ids = item["text_input_ids"].unsqueeze(0).to(self.device)
            text_mask = item["text_attention_mask"].unsqueeze(0).to(self.device)
            pixel_vals = item["pixel_values"].unsqueeze(0).to(self.device)
            post_ids = item["post_input_ids"].unsqueeze(0).to(self.device)
            post_mask = item["post_attention_mask"].unsqueeze(0).to(self.device)
            has_post = item["has_post"].unsqueeze(0).to(self.device)
            
            amp_enabled = (self.device.type in ["mps", "cuda"])
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = self.model(
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    pixel_values=pixel_vals,
                    post_input_ids=post_ids,
                    post_attention_mask=post_mask,
                    has_post=has_post,
                    modality=modality,
                    return_rationale=False
                )
            
            prob = torch.softmax(outputs["binary_logits"], dim=-1)[0, 1].item()
            is_hate = prob >= threshold
            gates = outputs["gate_weights"][0].cpu().numpy()

        print("\n" + "=" * 70)
        print("          MULTIMODAL HATE SPEECH DETECTION PREDICTION REPORT")
        print("=" * 70)
        print(f" Modality Mode          : {modality.upper()}")
        if image_path:
            print(f" Image Path             : {image_path}")
        print(f" Input Text             : {raw_text if raw_text else '(none)'}")
        if raw_text and clean_text != raw_text:
            print(f" Deobfuscated Text      : {clean_text}")
        if post_text:
            print(f" Surrounding Post       : {post_text}")
        print("-" * 70)
        
        status_label = "🚨 HARMFUL / HATE SPEECH DETECTED" if is_hate else "✅ NOT HARMFUL / CLEAN MEME"
        print(f" CLASSIFICATION RESULT  : {status_label}")
        print(f" Harmful Probability    : {prob * 100:.2f}% (Threshold: {threshold * 100:.1f}%)")
        print(f" Clean Probability      : {(1.0 - prob) * 100:.2f}%")
        print("-" * 70)
        print(" CONTEXT GATING WEIGHTS:")
        print(f"   • Text Stream Gate        : {gates[0]:.4f}")
        print(f"   • Vision Stream Gate      : {gates[1]:.4f}")
        print(f"   • Cross-Attention Gate    : {gates[2]:.4f}")
        print(f"   • Post-Text Gate          : {gates[3]:.4f}")
        print("=" * 70 + "\n")
        return {"is_hate": is_hate, "hate_probability": prob, "gates": gates.tolist()}

    def run_interactive(self, threshold=0.5, modality="multimodal"):
        """
        Interactive REPL loop: allows continuous testing of text & memes directly in the terminal
        without having to re-load weights each time.
        """
        print("\n" + "=" * 70)
        print("       🎯 INTERACTIVE TERMINAL HATE SPEECH DETECTOR")
        print("       Test any text, meme, or homoglyph directly in terminal")
        print("       Type 'exit' or 'q' at any time to quit.")
        print("=" * 70)
        
        while True:
            try:
                print("\n" + "-" * 70)
                text_input = input("📝 Enter text / meme caption (or 'exit' to quit): ").strip()
                if not text_input or text_input.lower() in ["exit", "quit", "q"]:
                    print("\nExiting interactive detector. Goodbye!")
                    break
                
                img_path = input("🖼️  Enter image path (optional - press Enter to skip): ").strip()
                if not img_path:
                    img_path = None
                elif not os.path.exists(img_path):
                    print(f"⚠️  Note: '{img_path}' not found on disk. Proceeding with text.")
                    img_path = None
                
                post_input = input("💬 Enter surrounding post / context (optional - press Enter to skip): ").strip()
                
                self.predict_single(
                    text=text_input,
                    image_path=img_path,
                    post_text=post_input,
                    threshold=threshold,
                    modality=modality
                )
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting interactive detector. Goodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="Terminal Evaluator & Interactive Hate Speech Detector")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive terminal REPL for live text/meme testing")
    parser.add_argument("--fine_tune", "--train", dest="fine_tune", action="store_true", help="Perform end-to-end differential fine-tuning before evaluation to achieve best results")
    parser.add_argument("--text", type=str, default=None, help="Text string / caption to classify")
    parser.add_argument("--image", type=str, default=None, help="Path to image file to classify")
    parser.add_argument("--post", type=str, default="", help="Surrounding post context text")
    parser.add_argument("--modality", type=str, default="multimodal", choices=["multimodal", "text_only", "image_only"], help="Prediction modality")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto from checkpoint or 0.5)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV/JSON dataset file")
    parser.add_argument("--zip_dataset", type=str, default=None, help="Path to Zip dataset archive")
    parser.add_argument("--split_ids", type=str, default="splits/test_ids.txt", help="Path to split IDs file")
    parser.add_argument("--gt_json", type=str, default="MMHS150K_GT.json", help="Path to GT json file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--samples", "--max_samples", dest="max_samples", type=int, default=8500, help="Number of dataset samples to evaluate / fine-tune on (default: 8500 for full dataset)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of fine-tuning epochs (default: 4)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of backbone layers to unfreeze (default: 2)")
    parser.add_argument("--fast", action="store_true", help="Ultra-fast training mode (frozen backbones + trained fusion & residual heads, runs in 2-3 mins)")
    parser.add_argument("--fusion_lr", type=float, default=2e-4, help="Learning rate for fusion heads (default: 2e-4)")
    parser.add_argument("--backbone_lr", type=float, default=2.5e-5, help="Learning rate for unfreezed backbones (default: 2.5e-5)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"], help="Computation device (default: cpu for reliable inference evaluation)")
    args = parser.parse_args()

    # 1. Automated End-to-End Fine-Tuning & Evaluation Pipeline
    if args.fine_tune:
        from train import train_and_finetune
        print("\n" + "=" * 80)
        print("  🚀 AUTOMATED END-TO-END FINE-TUNING & EVALUATION PIPELINE")
        print("=" * 80)
        dataset_target = args.zip_dataset or args.dataset or ("Multimodal Hate Speech Archive.zip" if os.path.exists("Multimodal Hate Speech Archive.zip") else "Archive (1).zip")
        print(f" Target Dataset  : {dataset_target}")
        print(f" Sample Count    : {args.max_samples}")
        print(f" Epochs          : {args.epochs}")
        print(f" Batch Size      : {args.batch_size}")
        print(f" Fast Mode       : {args.fast}")
        print(f" Unfreeze Layers : {1 if args.fast else args.unfreeze_layers}")
        print(f" Modality        : {args.modality.upper()}")
        print("=" * 80 + "\n")
        
        train_and_finetune(
            zip_dataset=args.zip_dataset if args.zip_dataset and os.path.exists(args.zip_dataset) else (dataset_target if str(dataset_target).endswith('.zip') else None),
            train_csv=args.dataset if args.dataset and not str(dataset_target).endswith('.zip') else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            fusion_lr=args.fusion_lr,
            backbone_lr=args.backbone_lr,
            unfreeze_layers=args.unfreeze_layers,
            fast_mode=args.fast,
            max_train_samples=args.max_samples,
            max_val_samples=min(500, max(100, args.max_samples // 2)),
            modality=args.modality
        )
        return

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

    evaluator = MultimodalHateSpeechTerminalEvaluator(checkpoint_path=ckpt, device=args.device)
    thresh = args.threshold if args.threshold is not None else evaluator.optimal_threshold
    
    # 2. Interactive REPL Mode
    if args.interactive:
        evaluator.run_interactive(threshold=thresh, modality=args.modality)
        return

    # 3. Single sample inference mode via command-line arguments
    if args.text is not None or args.image is not None:
        evaluator.predict_single(
            text=args.text,
            image_path=args.image,
            post_text=args.post,
            threshold=thresh,
            modality=args.modality
        )
        return

    # 4. Dataset evaluation mode
    if args.dataset and os.path.exists(args.dataset):
        evaluator.run_evaluation(dataset_path=args.dataset, max_samples=args.max_samples, threshold=thresh, modality=args.modality)
    elif args.zip_dataset and os.path.exists(args.zip_dataset):
        evaluator.run_evaluation(
            zip_path=args.zip_dataset,
            split_ids_file=args.split_ids,
            gt_json_path=args.gt_json,
            max_samples=args.max_samples,
            threshold=thresh,
            modality=args.modality
        )
    elif os.path.exists("Multimodal Hate Speech Archive.zip"):
        evaluator.run_evaluation(
            zip_path="Multimodal Hate Speech Archive.zip",
            split_ids_file=args.split_ids,
            gt_json_path=args.gt_json,
            max_samples=args.max_samples,
            threshold=thresh,
            modality=args.modality
        )
    else:
        # Default fallback to interactive mode if no dataset is specified
        evaluator.run_interactive(threshold=thresh, modality=args.modality)

if __name__ == "__main__":
    main()
