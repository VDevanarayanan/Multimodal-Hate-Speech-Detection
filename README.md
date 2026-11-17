# Multimodal Multilingual Hate Speech Detection System

An advanced multimodal hate speech detection framework built with **Dual-Directional Cross-Attention Fusion (DDCAF)** and **Dynamic Context Gating**. The system addresses key challenges in existing multimodal architectures by enabling dynamic cross-modal alignment, 100+ language multilingual support, defense against cloaking/obfuscation tricks, and robust performance in low-data regimes.

---

## 🌟 Key Features & Architectural Innovations

1. **Dual-Directional Cross-Attention Fusion (DDCAF Core)**:
   - **Text-Guided Visual Attention (TGVA)**: Text tokens attend to visual patch embeddings.
   - **Visual-Guided Text Attention (VGTA)**: Visual patches attend to text tokens to capture fine-grained spatial-semantic interactions.
2. **Dynamic Context Gating Fusion**:
   - Learns adaptive gating weights across visual features, caption/OCR text, and cross-attention representations while dynamically suppressing noisy surrounding post text.
3. **Multilingual Backbone (100+ Languages)**:
   - Powered by **XLM-RoBERTa** (`xlm-roberta-base`) for both caption/OCR text and surrounding post text, paired with **ViT** (`google/vit-base-patch16-224-in21k`) for visual patch representation.
4. **Cloaking & Obfuscation Defense (`TextDeobfuscator`)**:
   - Resolves Cyrillic/Greek homoglyphs, leetspeak (`h@t3` $\rightarrow$ `hate`, `r$c!st` $\rightarrow$ `racist`), zero-width spaces, and character insertion delimiters (`r.a.c.i.s.t` $\rightarrow$ `racist`).
5. **Multi-Task Learning Heads**:
   - **Binary Head**: Hate Speech vs. Normal / Non-hate.
   - **Multi-Label Head**: 8 Hate categories (Racism/Xenophobia, Sexism/Misogyny, Homophobia/LGBTQ+, Religious Hate, Disability Discrimination, Political Extremism, Cyberbullying/Threat, Dehumanization/Slurs).
   - **Rationale Head**: Verb-object phrase / rationale token extraction pointer.
6. **Low-Data & Imbalanced Data Optimization**:
   - Implements **Balanced Focal Loss** and parameter-efficient adapter fine-tuning to prevent majority class collapse on small datasets.
7. **Comprehensive Evaluation Metrics**:
   - Computes **Accuracy**, **Precision**, **Recall**, **F1-Score**, **AUC / ROC**, **Confusion Matrix**, and **Confidence Score Matrix** directly in the terminal or via a Web Dashboard.

---

## 🏗️ Architecture Overview

```text
                                [ Meme Input ]
                     (Image + Caption + Surrounding Post)
                                      │
        ┌───────────────┬─────────────┴─────────────┬──────────────┐
        ▼               ▼                           ▼              ▼
  [ OCR Extract ] [ RoBERTa Text ]            [ ViT Visual ] [ Post RoBERTa ]
    Image Text     Caption + OCR              Image Patches     Post Text
        └───────┬───────┘                           │              │
                └───────────────┬───────────────────┘              │
                                ▼                                  │
               [ Bidirectional Cross-Attention ]                   │
                   TGVA + VGTA Core (DDCAF)                        │
                                │                                  │
                                └─────────────────┬────────────────┘
                                                  ▼
                                      [ Dynamic Gating Fusion ]
                                 Context Gate Suppresses Noisy Post
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
          [ Binary Head ]                [ Multi-Label Head ]              [ Rationale Head ]
           Hate / Normal                  8 Hate Categories                Verb-Object Phrase
                 │                                │                                │
                 └────────────────────────────────┼────────────────────────────────┘
                                                  ▼
                                           [ Final Output ]
```

---

## 📦 Project Structure

```text
├── config.py                 # Hyperparameters, hate categories, backbone settings
├── dataset_loader.py         # Multimodal dataset loader (CSV, JSON, and Zip archives)
├── evaluate.py               # Dataset evaluation engine computing Accuracy, F1, AUC/ROC
├── run_detection.py          # Standalone terminal evaluation runner with formatted tables
├── text_deobfuscator.py      # Unicode homoglyph, leetspeak, and zero-width cleaner
├── ocr_extractor.py          # OCR extraction engine (EasyOCR / PyTesseract / PIL)
├── train.py                  # Fine-tuning engine with Balanced Focal Loss
├── app.py                    # Interactive Web Dashboard UI and API server
├── models/
│   ├── ddcaf_fusion.py       # Bidirectional cross-attention (TGVA+VGTA) & Context Gate
│   ├── vit_stream.py         # Vision Transformer (ViT) patch extractor
│   ├── xlm_roberta_stream.py # Multilingual XLM-RoBERTa encoder
│   ├── multitask_heads.py    # Binary, Multi-label, and Rationale classification heads
│   └── multimodal_hate_speech_model.py # End-to-end model pipeline
├── sample_data/              # Sample multilingual test benchmarks (CSV / JSON)
├── splits/                   # Train / Val / Test ID splits
└── requirements.txt          # Python dependencies
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/VDevanarayanan/Multimodal-Hate-Speech-Detection.git
cd Multimodal-Hate-Speech-Detection

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. End-to-End Automated Fine-Tuning & Evaluation (Best Performance)

To automatically fine-tune with differential learning rates, automatically calibrate the optimal decision threshold, and output the best possible evaluation metrics in the terminal:

```bash
# On HarMeme Dataset (Full COVID-19 Harmful Memes):
python train.py --zip_dataset harmeme.zip

# On Facebook Hateful Memes:
python run_detection.py --zip_dataset "Archive (1).zip" --samples 1000 --fine_tune

# On MMHS150K Twitter Dataset:
python run_detection.py --zip_dataset "Multimodal Hate Speech Archive.zip" --samples 1000 --fine_tune
```

### 3. Run Evaluation on Existing Checkpoint

Running detection automatically detects and loads the best saved model (`checkpoints/best_model.pt`) and its calibrated decision threshold:

```bash
# Evaluate on HarMeme test set (auto-loads best_model.pt & calibrated threshold):
python evaluate.py --zip_dataset harmeme.zip --split_ids test.jsonl

# Or with run_detection.py terminal summary tables:
python run_detection.py --zip_dataset harmeme.zip --split_ids test.jsonl

# Evaluate on Facebook Hateful Memes:
python run_detection.py --zip_dataset "Archive (1).zip" --samples 500

# Evaluate on MMHS150K Twitter Dataset:
python run_detection.py --zip_dataset "Multimodal Hate Speech Archive.zip" --samples 500
```

### 4. Interactive Detection & Single Sample Inference

```bash
# Single sample classification (Text + Image):
python run_detection.py --text "Look at this group of people" --image "sample_data/meme1.jpg"

# Interactive REPL mode:
python run_detection.py -i
```

#### Terminal Output Preview:
```text
================================================================================
  🎯 MULTIMODAL HATE SPEECH DETECTION MODEL PERFORMANCE REPORT
================================================================================
 Total Evaluated Dataset Samples : 1000
 Classification Threshold         : 0.50
================================================================================

 📌 1. PRIMARY METRICS SUMMARY:
 ┌──────────────────────┬─────────────────┬─────────────────┐
 │ Metric               │ Percentage (%)  │ Decimal Score   │
 ├──────────────────────┼─────────────────┼─────────────────┤
 │ ACCURACY             │         71.50%  │          0.7150 │
 │ PRECISION (Macro)    │         49.84%  │          0.4984 │
 │ RECALL (Macro)       │         49.97%  │          0.4997 │
 │ F1-SCORE (Macro)     │         45.17%  │          0.4517 │
 │ AUC / ROC            │                 │          0.5193 │
 └──────────────────────┴─────────────────┴─────────────────┘

 🧩 2. CONFUSION MATRIX:
 ┌──────────────────────────────────┬──────────────────────────────────┐
 │                                  │          PREDICTED CLASS         │
 │ ACTUAL CLASS                     ├─────────────────┬────────────────┤
 │                                  │ Normal (0)      │ Hate Speech (1)│
 ├──────────────────────────────────┼─────────────────┼────────────────┤
 │ Actual Normal (0)                │ TN = 11         │ FP = 254       │
 │ Actual Hate Speech (1)           │ FN = 31         │ TP = 704       │
 └──────────────────────────────────┴─────────────────┴────────────────┘

 ⚡ 3. DYNAMIC CONTEXT GATING WEIGHTS:
   • Main Text Stream Gate (Caption + OCR) : 0.4439
   • ViT Visual Stream Gate (Image)        : 0.5664
   • DDCAF Cross-Attention Gate            : 0.3099
   • Post Noise Suppression Gate           : 0.3643
```

### 3. Fine-Tuning & Diagnostics

#### Diagnostic Sanity Check (Overfitting Test)
Verify that the model architecture, gradients, and loss functions can overfit 64 samples to ~100% training accuracy:
```bash
python train.py --overfit_check --max_train_samples 64
```

#### Differential Fine-Tuning
Fine-tune with unfrozen top transformer layers using differential learning rates:
```bash
# Fine-tune with top 2 layers unfrozen (backbone_lr=1.5e-5, fusion_lr=1e-4)
python train.py --epochs 3 --batch_size 4 --unfreeze_layers 2 --fusion_lr 1e-4 --backbone_lr 1.5e-5 --max_train_samples 1000

# Train binary head alone (without multi-label interference)
python train.py --binary_only --epochs 3 --max_train_samples 1000
```

#### Modality Baselines (Ablation Studies)
Isolate unimodal text vs image performance against full multimodal DDCAF:
```bash
# Text-Only Baseline (XLM-RoBERTa + Head)
python evaluate.py --zip_dataset "Multimodal Hate Speech Archive.zip" --modality text_only --max_samples 1000

# Image-Only Baseline (ViT + Head)
python evaluate.py --zip_dataset "Multimodal Hate Speech Archive.zip" --modality image_only --max_samples 1000

# Full Multimodal (DDCAF + Dynamic Gating)
python evaluate.py --zip_dataset "Multimodal Hate Speech Archive.zip" --modality multimodal --max_samples 1000
```

### 4. Interactive Web Dashboard (Optional)

If you prefer a browser dashboard:

```bash
python app.py
```
Then navigate to `http://127.0.0.1:5050` in your web browser.

---

## 🧪 Unit Tests

Run the test suite to verify module integrity:
```bash
python test_system.py
```

---

## 📜 License

MIT License. See `LICENSE` for details.
