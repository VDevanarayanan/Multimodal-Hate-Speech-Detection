import os
import io
import json
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from text_deobfuscator import TextDeobfuscator
from ocr_extractor import OCRExtractor
from config import CONFIG

class MultimodalHateSpeechDataset(Dataset):
    """
    Multimodal Multilingual Hate Speech Dataset class.
    Supports CSV, JSON, and direct reading from zip files like MMHS150K Archive.
    """
    def __init__(self, data_source, zip_path=None, split_ids_file=None, gt_json_path=None, tokenizer=None, ocr_extractor=None, max_len=128, max_samples=None):
        self.max_len = max_len
        self.zip_path = zip_path
        self.zip_file = None
        
        # Initialize Tokenizer & OCR
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.text_model_name)
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
            
        self.ocr_extractor = ocr_extractor if ocr_extractor is not None else OCRExtractor()
        
        # Image Transform
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load samples
        if zip_path and os.path.exists(zip_path) and gt_json_path and os.path.exists(gt_json_path):
            self.samples = self._load_from_zip(zip_path, split_ids_file, gt_json_path, max_samples)
        else:
            self.samples = self._load_samples(data_source)
            if max_samples and len(self.samples) > max_samples:
                self.samples = self.samples[:max_samples]

    def _load_from_zip(self, zip_path, split_ids_file, gt_json_path, max_samples=None):
        print(f"Loading dataset from zip archive: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            namelist = set(z.namelist())
            if 'data/dev.jsonl' in namelist or 'data/train.jsonl' in namelist:
                print("Detected Facebook Hateful Memes format in zip archive.")
                samples = []
                jsonl_files = [f for f in ['data/dev.jsonl', 'data/train.jsonl'] if f in namelist]
                for jf in jsonl_files:
                    lines = z.read(jf).decode('utf-8').strip().split('\n')
                    for line in lines:
                        if not line.strip(): continue
                        item = json.loads(line)
                        if 'label' not in item: continue
                        samples.append({
                            "id": str(item.get("id")),
                            "tweet_id": str(item.get("id")),
                            "caption": item.get("text", ""),
                            "img_zip_path": "data/" + item.get("img", ""),
                            "post_text": "",
                            "label": int(item.get("label", 0)),
                            "categories": [0.0] * CONFIG.num_categories,
                            "in_zip": True,
                            "is_hateful_memes": True
                        })
                if max_samples and max_samples < len(samples):
                    samples = samples[:max_samples]
                print(f"Loaded {len(samples)} dataset samples from Hateful Memes split.")
                return samples

        if gt_json_path and os.path.exists(gt_json_path):
            with open(gt_json_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        else:
            gt_data = {}
            
        selected_ids = []
        if split_ids_file and os.path.exists(split_ids_file):
            with open(split_ids_file, 'r', encoding='utf-8') as f:
                selected_ids = [line.strip() for line in f if line.strip()]
        else:
            selected_ids = list(gt_data.keys())
            
        if max_samples and max_samples < len(selected_ids):
            # Stratified sampling across classes to ensure balanced representation
            pos_ids = [tid for tid in selected_ids if tid in gt_data and sum(1 for l in gt_data[tid].get("labels", []) if l > 0) >= 2]
            neg_ids = [tid for tid in selected_ids if tid in gt_data and sum(1 for l in gt_data[tid].get("labels", []) if l > 0) < 2]
            half = max_samples // 2
            selected_ids = pos_ids[:half] + neg_ids[:half]
            import random
            random.seed(42)
            random.shuffle(selected_ids)
            
        samples = []
        for tweet_id in selected_ids:
            if tweet_id not in gt_data:
                continue
            item = gt_data[tweet_id]
            labels = item.get("labels", [0, 0, 0])
            
            # Majority vote: 1 if >= 2 annotators labeled hate (1-5), or >=1 annotator
            hate_count = sum(1 for l in labels if l > 0)
            binary_label = 1 if hate_count >= 2 else (1 if hate_count >= 1 else 0)
            
            # Map 6 classes to 8 categories multi-hot
            m_hot = [0.0] * CONFIG.num_categories
            for l in labels:
                if l == 1: m_hot[0] = 1.0 # Racist
                elif l == 2: m_hot[1] = 1.0 # Sexist
                elif l == 3: m_hot[2] = 1.0 # Homophobe
                elif l == 4: m_hot[3] = 1.0 # Religion
                elif l == 5: m_hot[7] = 1.0 # OtherHate -> Slurs / Dehumanization
                
            samples.append({
                "id": str(tweet_id),
                "tweet_id": str(tweet_id),
                "caption": item.get("tweet_text", ""),
                "post_text": "",
                "label": binary_label,
                "categories": m_hot,
                "in_zip": True
            })
            
        print(f"Loaded {len(samples)} dataset samples from split.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pixel_values = torch.zeros(3, 224, 224)
        extracted_ocr = item.get("ocr_text", "")
        
        # Check if loading from zip archive
        if item.get("in_zip", False) and self.zip_path:
            if self.zip_file is None:
                self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            
            if item.get("is_hateful_memes", False):
                img_zip_name = item.get("img_zip_path")
                if img_zip_name and img_zip_name in self.zip_file.namelist():
                    try:
                        img_bytes = self.zip_file.read(img_zip_name)
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        pixel_values = self.img_transform(pil_img)
                    except Exception:
                        pass
            else:
                tweet_id = item["tweet_id"]
                img_zip_name = f"img_resized/{tweet_id}.jpg"
                if img_zip_name in self.zip_file.namelist():
                    try:
                        img_bytes = self.zip_file.read(img_zip_name)
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        pixel_values = self.img_transform(pil_img)
                    except Exception:
                        pass
                        
                ocr_zip_name = f"img_txt/{tweet_id}.json"
                if ocr_zip_name in self.zip_file.namelist():
                    try:
                        ocr_bytes = self.zip_file.read(ocr_zip_name)
                        ocr_json = json.loads(ocr_bytes.decode('utf-8'))
                        if isinstance(ocr_json, list):
                            extracted_ocr = " ".join([o.get("text", "") for o in ocr_json if isinstance(o, dict)])
                    except Exception:
                        pass

        # 2. De-obfuscation & Text Cleaning
        caption_clean = TextDeobfuscator.clean(item.get("caption", ""))
        ocr_clean = TextDeobfuscator.clean(extracted_ocr)
        post_clean = TextDeobfuscator.clean(item.get("post_text", ""))
        
        # Combine Caption + OCR Text
        combined_text = f"{caption_clean} [SEP] {ocr_clean}" if ocr_clean else caption_clean
        if not combined_text.strip():
            combined_text = "multimodal content"
            
        # 3. Tokenize
        if self.tokenizer is not None:
            text_enc = self.tokenizer(
                combined_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            text_ids = text_enc["input_ids"].squeeze(0)
            text_mask = text_enc["attention_mask"].squeeze(0)
            
            post_enc = self.tokenizer(
                post_clean if post_clean else "none",
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            post_ids = post_enc["input_ids"].squeeze(0)
            post_mask = post_enc["attention_mask"].squeeze(0)
        else:
            text_ids = torch.tensor([ord(c) % 250000 for c in combined_text[:self.max_len]] + [0] * max(0, self.max_len - len(combined_text)))
            text_mask = torch.tensor([1 if i < len(combined_text) else 0 for i in range(self.max_len)])
            post_ids = torch.tensor([ord(c) % 250000 for c in post_clean[:self.max_len]] + [0] * max(0, self.max_len - len(post_clean)))
            post_mask = torch.tensor([1 if i < len(post_clean) else 0 for i in range(self.max_len)])

        # 4. Binary & Multi-Label Targets
        binary_label = torch.tensor(int(item.get("label", 0)), dtype=torch.long)
        
        cats = item.get("categories", [0] * CONFIG.num_categories)
        if isinstance(cats, list) and len(cats) == CONFIG.num_categories:
            multilabel_target = torch.tensor(cats, dtype=torch.float32)
        else:
            m_hot = [0.0] * CONFIG.num_categories
            if isinstance(cats, list):
                for c in cats:
                    if 0 <= c < CONFIG.num_categories:
                        m_hot[c] = 1.0
            multilabel_target = torch.tensor(m_hot, dtype=torch.float32)

        return {
            "id": item.get("id", str(idx)),
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "pixel_values": pixel_values,
            "post_input_ids": post_ids,
            "post_attention_mask": post_mask,
            "binary_label": binary_label,
            "multilabel_target": multilabel_target,
            "raw_text": combined_text,
            "raw_post": post_clean
        }

    def _load_samples(self, data_source):
        samples = []
        if isinstance(data_source, list):
            return data_source
        elif isinstance(data_source, str):
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                for _, row in df.iterrows():
                    samples.append({
                        "id": row.get("id", str(_)),
                        "image_path": row.get("image_path", row.get("image", "")),
                        "caption": str(row.get("caption", row.get("text", ""))),
                        "post_text": str(row.get("post_text", row.get("surrounding_text", ""))),
                        "ocr_text": str(row.get("ocr_text", "")),
                        "label": int(row.get("label", row.get("hate_label", 0))),
                        "categories": json.loads(row["categories"]) if isinstance(row.get("categories"), str) and row.get("categories").startswith("[") else row.get("categories", [0] * CONFIG.num_categories)
                    })
            elif data_source.endswith('.json'):
                with open(data_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    elif isinstance(data, dict) and "data" in data:
                        samples = data["data"]
        return samples
