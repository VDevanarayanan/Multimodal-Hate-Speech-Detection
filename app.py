import os
import json
import tempfile
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

from config import CONFIG
from evaluate import Evaluator
from text_deobfuscator import TextDeobfuscator
from models.multimodal_hate_speech_model import MultimodalHateSpeechModel

app = Flask(__name__)

# Initialize evaluator & model
evaluator = Evaluator()

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Multilingual Hate Speech Detection & Evaluation System</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-card: rgba(18, 24, 38, 0.75);
            --border-glow: rgba(99, 102, 241, 0.25);
            --accent-purple: #6366f1;
            --accent-teal: #14b8a6;
            --accent-coral: #f43f5e;
            --accent-gold: #f59e0b;
            --accent-blue: #3b82f6;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(20, 184, 166, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(244, 63, 94, 0.08) 0%, transparent 60%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            max-width: 1240px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #14b8a6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        p.subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
        }

        .badge-row {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .badge {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #c7d2fe;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 500;
        }

        .badge.teal {
            background: rgba(20, 184, 166, 0.15);
            border-color: rgba(20, 184, 166, 0.3);
            color: #99f6e4;
        }

        .badge.coral {
            background: rgba(244, 63, 94, 0.15);
            border-color: rgba(244, 63, 94, 0.3);
            color: #fecdd3;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 850px) {
            .grid-2 { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .btn {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }

        input[type="text"], select {
            width: 100%;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: var(--text-primary);
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.92rem;
            outline: none;
        }

        /* 5 Metric Cards Row */
        .metrics-cards-5 {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        @media (max-width: 900px) {
            .metrics-cards-5 { grid-template-columns: repeat(2, 1fr); }
        }

        .metric-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.25rem 0.8rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .metric-box::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
        }

        .metric-box.acc::before { background: var(--accent-blue); }
        .metric-box.prec::before { background: var(--accent-teal); }
        .metric-box.rec::before { background: var(--accent-gold); }
        .metric-box.f1::before { background: var(--accent-purple); }
        .metric-box.roc::before { background: var(--accent-coral); }

        .metric-label {
            font-size: 0.78rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .metric-val {
            font-family: 'Outfit', sans-serif;
            font-size: 2rem;
            font-weight: 800;
        }

        .metric-box.acc .metric-val { color: #60a5fa; }
        .metric-box.prec .metric-val { color: #2dd4bf; }
        .metric-box.rec .metric-val { color: #fbbf24; }
        .metric-box.f1 .metric-val { color: #818cf8; }
        .metric-box.roc .metric-val { color: #f43f5e; }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 0.88rem;
        }

        th {
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.1);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            width: 80px;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #14b8a6);
            border-radius: 4px;
        }

        .deobfuscated-box {
            background: rgba(99, 102, 241, 0.08);
            border-left: 3px solid var(--accent-purple);
            padding: 10px 14px;
            font-size: 0.88rem;
            margin-top: 0.75rem;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Multimodal Multilingual Hate Speech Detection</h1>
            <p class="subtitle">Evaluator & Detector powered by DDCAF Cross-Attention & Dynamic Context Gate</p>
            <div class="badge-row">
                <span class="badge teal">Dynamic Fusion (DDCAF)</span>
                <span class="badge">MMHS150K Dataset Support</span>
                <span class="badge coral">Accuracy, Precision, Recall, F1, AUC/ROC</span>
                <span class="badge teal">Cloaking Defense</span>
            </div>
        </header>

        <div class="grid-2">
            <!-- Dataset Evaluation Card -->
            <div class="card">
                <div class="card-title">
                    <span class="icon">📦</span> Evaluate Dataset Archive (.zip / .csv / .json)
                </div>
                <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 1.25rem;">
                    Run evaluation on <strong>Multimodal Hate Speech Archive.zip (MMHS150K)</strong> or custom dataset files to compute <strong>Accuracy, Precision, Recall, F1 Score, and AUC/ROC</strong>.
                </p>

                <div class="form-group">
                    <label>Select Dataset Target:</label>
                    <select id="datasetSelect">
                        <option value="zip_archive">Multimodal Hate Speech Archive.zip (MMHS150K Test Split)</option>
                        <option value="sample_data/multilingual_hate_speech_benchmark.csv">Multilingual Benchmark CSV</option>
                        <option value="sample_data/multilingual_hate_speech_benchmark.json">Multilingual Benchmark JSON</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Sample Limit (Max samples to evaluate, e.g. 500 or Full):</label>
                    <input type="text" id="maxSamples" value="500" placeholder="Enter max samples or leave blank for all">
                </div>

                <button class="btn" onclick="runDatasetEvaluation()">
                    🚀 Calculate Accuracy, Precision, Recall, F1 & AUC/ROC
                </button>
            </div>

            <!-- Single Meme Detector Card -->
            <div class="card">
                <div class="card-title">
                    <span class="icon">🔍</span> Instant Meme Detector & De-obfuscator
                </div>
                <div class="form-group">
                    <label>Caption / Text (Supports Obfuscated / Leetspeak Text):</label>
                    <input type="text" id="memeText" value="G3t0ut of my country r@c!st h@t3r" placeholder="Enter meme text or caption...">
                </div>
                <div class="form-group">
                    <label>Surrounding Post Text (Optional):</label>
                    <input type="text" id="postText" value="Breaking news debate thread on social media" placeholder="Enter post text...">
                </div>
                <button class="btn" onclick="predictSingleMeme()">
                    ⚡ Predict Hate & De-obfuscate
                </button>

                <div id="singleResult" style="margin-top: 1.25rem; display: none;"></div>
            </div>
        </div>

        <!-- Evaluation Results Display -->
        <div id="evalResultsSection" style="display: none;">
            <div class="card" style="margin-bottom: 2rem;">
                <div class="card-title">
                    <span class="icon">🏆</span> Comprehensive Dataset Performance Metrics
                </div>

                <div class="metrics-cards-5">
                    <div class="metric-box acc">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-val" id="metricAcc">--%</div>
                    </div>
                    <div class="metric-box prec">
                        <div class="metric-label">Precision</div>
                        <div class="metric-val" id="metricPrecision">--%</div>
                    </div>
                    <div class="metric-box rec">
                        <div class="metric-label">Recall</div>
                        <div class="metric-val" id="metricRecall">--%</div>
                    </div>
                    <div class="metric-box f1">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-val" id="metricF1">--%</div>
                    </div>
                    <div class="metric-box roc">
                        <div class="metric-label">AUC / ROC</div>
                        <div class="metric-val" id="metricAUC">--</div>
                    </div>
                </div>

                <div style="margin-top: 1.5rem;">
                    <h3 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem;">Dynamic Gating Fusion Weights</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
                        Learned gating weights dynamically suppressing noisy surrounding post text while balancing visual and cross-attention streams:
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;" id="gatesContainer"></div>
                </div>

                <div style="margin-top: 2rem;">
                    <h3 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem;">8-Category Hate Speech Breakdown</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Score Visualizer</th>
                            </tr>
                        </thead>
                        <tbody id="categoryTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runDatasetEvaluation() {
            const datasetSel = document.getElementById('datasetSelect').value;
            const maxSamp = document.getElementById('maxSamples').value;
            
            const payload = {
                max_samples: maxSamp ? parseInt(maxSamp) : null
            };
            if (datasetSel === 'zip_archive') {
                payload.zip_dataset = "Multimodal Hate Speech Archive.zip";
                payload.split_ids = "splits/test_ids.txt";
                payload.gt_json = "MMHS150K_GT.json";
            } else {
                payload.dataset = datasetSel;
            }

            const res = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                const pm = data.results.primary_metrics;
                document.getElementById('metricAcc').innerText = (pm.accuracy * 100).toFixed(1) + '%';
                document.getElementById('metricPrecision').innerText = (pm.precision * 100).toFixed(1) + '%';
                document.getElementById('metricRecall').innerText = (pm.recall * 100).toFixed(1) + '%';
                document.getElementById('metricF1').innerText = (pm.f1_score * 100).toFixed(1) + '%';
                document.getElementById('metricAUC').innerText = pm.auc_roc.toFixed(4);

                // Render Dynamic Gates
                const gates = data.results.average_dynamic_gates;
                const gatesHtml = `
                    <div class="metric-box"><div class="metric-label">Main Text</div><div class="metric-val" style="font-size:1.4rem">${gates.main_text_gate}</div></div>
                    <div class="metric-box"><div class="metric-label">Vision Stream</div><div class="metric-val" style="font-size:1.4rem">${gates.vision_gate}</div></div>
                    <div class="metric-box"><div class="metric-label">Cross-Attention</div><div class="metric-val" style="font-size:1.4rem">${gates.cross_attention_gate}</div></div>
                    <div class="metric-box"><div class="metric-label">Post Suppression</div><div class="metric-val" style="font-size:1.4rem; color:#f43f5e">${gates.post_text_suppression_gate}</div></div>
                `;
                document.getElementById('gatesContainer').innerHTML = gatesHtml;

                // Render Category Table
                let tableHtml = '';
                for (const [catName, metrics] of Object.entries(data.results.category_metrics)) {
                    const f1Pct = (metrics.f1_score * 100).toFixed(1);
                    tableHtml += `
                        <tr>
                            <td><strong>${catName}</strong></td>
                            <td>${(metrics.accuracy * 100).toFixed(1)}%</td>
                            <td>${(metrics.precision * 100).toFixed(1)}%</td>
                            <td>${(metrics.recall * 100).toFixed(1)}%</td>
                            <td><strong>${f1Pct}%</strong></td>
                            <td>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill" style="width: ${f1Pct}%"></div>
                                </div>
                            </td>
                        </tr>
                    `;
                }
                document.getElementById('categoryTableBody').innerHTML = tableHtml;
                document.getElementById('evalResultsSection').style.display = 'block';
            }
        }

        async function predictSingleMeme() {
            const text = document.getElementById('memeText').value;
            const postText = document.getElementById('postText').value;
            
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ caption: text, post_text: postText })
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                const isHate = data.prediction.is_hate;
                const pillStyle = isHate ? 'background:rgba(244, 63, 94, 0.2); color:#f43f5e; border:1px solid rgba(244, 63, 94, 0.4);' : 'background:rgba(20, 184, 166, 0.2); color:#14b8a6; border:1px solid rgba(20, 184, 166, 0.4);';
                const pillText = isHate ? '⚠️ HATE SPEECH DETECTED' : '✅ NORMAL / NON-HATE';
                
                let catsHtml = data.prediction.detected_categories.map(c => `<span class="badge coral">${c}</span>`).join(' ');
                if (!catsHtml) catsHtml = '<span class="badge teal">None</span>';

                const html = `
                    <div style="display:inline-block; padding:6px 16px; border-radius:20px; font-weight:700; margin-bottom:1rem; ${pillStyle}">${pillText}</div>
                    <div class="deobfuscated-box">
                        <strong>Un-cloaked & Normalized Text:</strong><br>
                        <em>${data.prediction.deobfuscated_text}</em>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.88rem;">
                        <strong>Hate Probability:</strong> ${(data.prediction.hate_probability * 100).toFixed(1)}%<br>
                        <strong>Detected Categories:</strong> ${catsHtml}
                    </div>
                `;
                const container = document.getElementById('singleResult');
                container.innerHTML = html;
                container.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/evaluate', methods=['POST'])
def evaluate_api():
    req = request.get_json() or {}
    dataset_path = req.get("dataset", None)
    zip_path = req.get("zip_dataset", None)
    split_ids = req.get("split_ids", "splits/test_ids.txt")
    gt_json = req.get("gt_json", "MMHS150K_GT.json")
    max_samples = req.get("max_samples", None)
    
    if zip_path or os.path.exists("Multimodal Hate Speech Archive.zip"):
        z_path = zip_path if zip_path else "Multimodal Hate Speech Archive.zip"
        results = evaluator.evaluate_dataset(
            zip_path=z_path,
            split_ids_file=split_ids,
            gt_json_path=gt_json,
            max_samples=max_samples
        )
    else:
        results = evaluator.evaluate_dataset(data_source=dataset_path, max_samples=max_samples)
        
    return jsonify({"status": "success", "results": results})

@app.route('/api/predict', methods=['POST'])
def predict_api():
    req = request.get_json() or {}
    caption = req.get("caption", "")
    post_text = req.get("post_text", "")
    
    clean_caption = TextDeobfuscator.clean(caption)
    
    sample = [{
        "id": "single_test",
        "caption": caption,
        "post_text": post_text,
        "ocr_text": "",
        "label": 0,
        "categories": [0]*CONFIG.num_categories
    }]
    
    dataset = MultimodalHateSpeechDataset(sample, max_len=CONFIG.max_text_len)
    item = dataset[0]
    
    with torch.no_grad():
        text_ids = item["text_input_ids"].unsqueeze(0).to(evaluator.device)
        text_mask = item["text_attention_mask"].unsqueeze(0).to(evaluator.device)
        pixel_vals = item["pixel_values"].unsqueeze(0).to(evaluator.device)
        post_ids = item["post_input_ids"].unsqueeze(0).to(evaluator.device)
        post_mask = item["post_attention_mask"].unsqueeze(0).to(evaluator.device)
        
        outputs = evaluator.model(
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            pixel_values=pixel_vals,
            post_input_ids=post_ids,
            post_attention_mask=post_mask
        )
        
        prob = torch.softmax(outputs["binary_logits"], dim=-1)[0, 1].item()
        ml_probs = torch.sigmoid(outputs["multilabel_logits"])[0].cpu().numpy()
        
        detected_cats = [CONFIG.hate_categories[i] for i, p in enumerate(ml_probs) if p >= 0.4]
        
    return jsonify({
        "status": "success",
        "prediction": {
            "is_hate": prob >= 0.5,
            "hate_probability": round(prob, 4),
            "deobfuscated_text": clean_caption,
            "detected_categories": detected_cats
        }
    })

if __name__ == '__main__':
    print("Starting Multimodal Multilingual Hate Speech Detection Web Dashboard at http://127.0.0.1:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
