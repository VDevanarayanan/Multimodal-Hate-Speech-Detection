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
    def __init__(self, data_source=None, zip_path=None, split_ids_file=None, gt_json_path=None, tokenizer=None, ocr_extractor=None, max_len=128, max_samples=None, load_images=True):
        self.max_len = max_len
        self.zip_path = zip_path
        self.zip_file = None
        self.zip_namelist = None
        self.load_images = load_images
        
        # Initialize Tokenizer & OCR
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.text_model_name)
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
            
        self.ocr_extractor = ocr_extractor if ocr_extractor is not None else OCRExtractor()
        
        # Image Transform - ViT direct (224, 224) resize for maximum throughput (mean/std = 0.5)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Token Cache & Pre-tokenized dummy values for zero-latency batch iterations
        self._token_cache = {}
        if self.tokenizer is not None:
            try:
                dummy_enc = self.tokenizer(
                    "none",
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                self._dummy_post_ids = dummy_enc["input_ids"].squeeze(0)
                self._dummy_post_mask = torch.zeros(self.max_len, dtype=torch.long)
            except Exception:
                self._dummy_post_ids = torch.zeros(self.max_len, dtype=torch.long)
                self._dummy_post_mask = torch.zeros(self.max_len, dtype=torch.long)
        else:
            self._dummy_post_ids = torch.zeros(self.max_len, dtype=torch.long)
            self._dummy_post_mask = torch.zeros(self.max_len, dtype=torch.long)
        
        # Load samples
        if zip_path and os.path.exists(zip_path):
            self.samples = self._load_from_zip(zip_path, split_ids_file, gt_json_path, max_samples)
        else:
            self.samples = self._load_samples(data_source)
            if max_samples and len(self.samples) > max_samples:
                self.samples = self.samples[:max_samples]
        
        is_full_dataset = any(s.get("is_harmeme") or s.get("is_multioff") or s.get("is_hateful_memes") for s in self.samples)
        if not is_full_dataset and max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _load_from_zip(self, zip_path, split_ids_file, gt_json_path, max_samples=None):
        print(f"Loading dataset from zip archive: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            namelist = set(z.namelist())
            if 'data/dev.jsonl' in namelist or 'data/train.jsonl' in namelist:
                print("Detected Facebook Hateful Memes format in zip archive.")
                samples = []
                # Distinguish dev vs train split to prevent mixing training and evaluation data
                is_train_split = split_ids_file and "train" in split_ids_file.lower()
                if is_train_split and 'data/train.jsonl' in namelist:
                    jsonl_files = ['data/train.jsonl']
                elif 'data/dev.jsonl' in namelist:
                    jsonl_files = ['data/dev.jsonl']
                else:
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
                            "zip_path": zip_path,
                            "is_hateful_memes": True,
                            "has_category_labels": False
                        })
                if max_samples and max_samples < len(samples):
                    # Balance positive and negative samples if subsampling
                    pos = [s for s in samples if s["label"] == 1]
                    neg = [s for s in samples if s["label"] == 0]
                    half = max_samples // 2
                    import random
                    random.seed(42)
                    samples = pos[:half] + neg[:half]
                    random.shuffle(samples)
                print(f"Loaded {len(samples)} dataset samples from Hateful Memes split ({jsonl_files}).")
                return samples

            # HarMeme (Harmful Memes) dataset format
            elif 'harmeme' in zip_path.lower() or ('train.jsonl' in namelist and 'vocabulary_100k.txt' in namelist):
                print("Detected HarMeme (Harmful Memes) dataset format in zip archive.")
                samples = []
                split_name = "train.jsonl"
                if split_ids_file:
                    s_lower = split_ids_file.lower()
                    if "val" in s_lower or "dev" in s_lower:
                        split_name = "val.jsonl"
                    elif "test" in s_lower:
                        split_name = "test.jsonl"
                    elif "train" in s_lower:
                        split_name = "train.jsonl"
                
                if split_name not in namelist:
                    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                        if f in namelist:
                            split_name = f
                            break

                lines = z.read(split_name).decode('utf-8').strip().split('\n')
                for line in lines:
                    if not line.strip(): continue
                    item = json.loads(line)
                    labels_list = item.get("labels", [])
                    is_harmful = 1 if any(l in ["somewhat harmful", "very harmful", "harmful"] for l in labels_list) else 0
                    
                    img_file = item.get("image", "")
                    if f"images/images/{img_file}" in namelist:
                        img_path = f"images/images/{img_file}"
                    elif f"images/{img_file}" in namelist:
                        img_path = f"images/{img_file}"
                    else:
                        img_path = img_file
                    
                    m_hot = [0.0] * CONFIG.num_categories
                    if is_harmful == 1:
                        for l in labels_list:
                            if l == "individual": m_hot[6] = 1.0     # Cyberbullying / Target: Individual
                            elif l == "community": m_hot[0] = 1.0    # Racism / Targeted Community
                            elif l == "society": m_hot[5] = 1.0      # Political Extremism / Society
                            elif l == "organization": m_hot[5] = 1.0 # Organization

                    samples.append({
                        "id": str(item.get("id")),
                        "tweet_id": str(item.get("id")),
                        "caption": item.get("text", ""),
                        "img_zip_path": img_path,
                        "post_text": "",
                        "label": is_harmful,
                        "categories": m_hot,
                        "in_zip": True,
                        "zip_path": zip_path,
                        "is_harmeme": True,
                        "has_category_labels": True
                    })

                print(f"Loaded {len(samples)} dataset samples from HarMeme split ({split_name}) (Harmful: {sum(1 for s in samples if s['label']==1)}, Non-harmful: {sum(1 for s in samples if s['label']==0)}). [100% Full Dataset]")
                return samples

            # MultiOFF (Offensive Memes) dataset format
            elif 'multioff' in zip_path.lower() or 'Training_meme_dataset.csv' in namelist:
                print("Detected MultiOFF (Offensive Memes) dataset format in zip archive.")
                import pandas as pd
                import io
                samples = []
                csv_file = "Training_meme_dataset.csv"
                if split_ids_file:
                    s_lower = split_ids_file.lower()
                    if "val" in s_lower or "dev" in s_lower:
                        csv_file = "Validation_meme_dataset.csv"
                    elif "test" in s_lower:
                        csv_file = "Testing_meme_dataset.csv"
                    elif "train" in s_lower:
                        csv_file = "Training_meme_dataset.csv"

                if csv_file not in namelist:
                    for f in ["Training_meme_dataset.csv", "Validation_meme_dataset.csv", "Testing_meme_dataset.csv"]:
                        if f in namelist:
                            csv_file = f
                            break

                df = pd.read_csv(io.BytesIO(z.read(csv_file)))
                for _, row in df.iterrows():
                    lbl_raw = str(row.get("label", row.get("Label", ""))).strip().lower()
                    is_off = 1 if lbl_raw in ["offensive", "1", "hate", "true"] else 0
                    img_name = str(row.get("image_name", row.get("image", ""))).strip()
                    sentence = str(row.get("sentence", row.get("text", ""))).strip()
                    if sentence == "nan": sentence = ""
                    
                    img_path = None
                    if img_name in namelist:
                        img_path = img_name
                    elif f"images/{img_name}" in namelist:
                        img_path = f"images/{img_name}"
                        
                    samples.append({
                        "id": img_name,
                        "tweet_id": img_name,
                        "caption": sentence,
                        "img_zip_path": img_path,
                        "post_text": "",
                        "label": is_off,
                        "categories": [0.0] * CONFIG.num_categories,
                        "in_zip": bool(img_path),
                        "zip_path": zip_path,
                        "is_multioff": True,
                        "has_category_labels": False
                    })

                print(f"Loaded {len(samples)} dataset samples from MultiOFF split ({csv_file}) (Offensive: {sum(1 for s in samples if s['label']==1)}, Non-offensive: {sum(1 for s in samples if s['label']==0)}). [100% Full Dataset]")
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
            # Stratified sampling: 50% true positive (>= 2 hate votes), 50% clean non-hate (< 2 hate votes)
            pos_ids = [tid for tid in selected_ids if tid in gt_data and sum(1 for l in gt_data[tid].get("labels", []) if l > 0) >= 2]
            neg_ids = [tid for tid in selected_ids if tid in gt_data and sum(1 for l in gt_data[tid].get("labels", []) if l > 0) < 2]
            half = min(len(pos_ids), len(neg_ids), max_samples // 2)
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
            
            # MMHS150K standard majority vote: Hateful only if >= 2 annotators agree on hate (labels 1-5)
            hate_count = sum(1 for l in labels if l > 0)
            binary_label = 1 if hate_count >= 2 else 0
            
            # Map 6 classes to 8 categories multi-hot (only if hateful)
            m_hot = [0.0] * CONFIG.num_categories
            if binary_label == 1:
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
                "in_zip": True,
                "zip_path": zip_path,
                "has_category_labels": True
            })
            
        print(f"Loaded {len(samples)} dataset samples from split (Hate: {sum(1 for s in samples if s['label'] == 1)}, Non-hate: {sum(1 for s in samples if s['label'] == 0)}).")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pixel_values = torch.zeros(3, 224, 224)
        has_image_bool = False
        extracted_ocr = item.get("ocr_text", "")
        
        # Check if loading from zip archive
        zip_p = item.get("zip_path", self.zip_path)
        if item.get("in_zip", False) and zip_p:
            if not hasattr(self, "_zip_handles"):
                self._zip_handles = {}
                self._zip_namelists = {}
            if zip_p not in self._zip_handles:
                self._zip_handles[zip_p] = zipfile.ZipFile(zip_p, 'r')
                try:
                    self._zip_namelists[zip_p] = set(self._zip_handles[zip_p].namelist())
                except Exception:
                    self._zip_namelists[zip_p] = None
            
            z_file = self._zip_handles[zip_p]
            z_namelist = self._zip_namelists[zip_p]
            
            if self.load_images and (item.get("is_hateful_memes", False) or item.get("is_harmeme", False) or item.get("is_multioff", False)):
                img_zip_name = item.get("img_zip_path")
                if img_zip_name and (z_namelist is None or img_zip_name in z_namelist):
                    try:
                        img_bytes = z_file.read(img_zip_name)
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        pixel_values = self.img_transform(pil_img)
                        has_image_bool = True
                    except Exception:
                        pass
            elif not (item.get("is_hateful_memes", False) or item.get("is_harmeme", False) or item.get("is_multioff", False)):
                tweet_id = item["tweet_id"]
                img_zip_name = f"img_resized/{tweet_id}.jpg"
                if self.load_images and (z_namelist is None or img_zip_name in z_namelist):
                    try:
                        img_bytes = z_file.read(img_zip_name)
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        pixel_values = self.img_transform(pil_img)
                        has_image_bool = True
                    except Exception:
                        pass
                        
                ocr_zip_name = f"img_txt/{tweet_id}.json"
                if z_namelist is None or ocr_zip_name in z_namelist:
                    try:
                        ocr_bytes = z_file.read(ocr_zip_name)
                        ocr_json = json.loads(ocr_bytes.decode('utf-8'))
                        if isinstance(ocr_json, dict):
                            extracted_ocr = ocr_json.get("img_text", "") or ocr_json.get("text", "")
                        elif isinstance(ocr_json, list):
                            extracted_ocr = " ".join([o.get("text", "") for o in ocr_json if isinstance(o, dict)])
                    except Exception:
                        pass
        elif self.load_images and item.get("image_path") and os.path.exists(item["image_path"]):
            try:
                pil_img = Image.open(item["image_path"]).convert('RGB')
                pixel_values = self.img_transform(pil_img)
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
            
        # 3. Fast Tokenization with In-Memory Caching
        if idx in self._token_cache:
            text_ids, text_mask, post_ids, post_mask, has_post_bool = self._token_cache[idx]
        else:
            has_post_bool = bool(post_clean.strip())
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
                
                if has_post_bool:
                    post_enc = self.tokenizer(
                        post_clean,
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    post_ids = post_enc["input_ids"].squeeze(0)
                    post_mask = post_enc["attention_mask"].squeeze(0)
                else:
                    post_ids = self._dummy_post_ids
                    post_mask = self._dummy_post_mask
            else:
                text_ids = torch.tensor([ord(c) % 250000 for c in combined_text[:self.max_len]] + [0] * max(0, self.max_len - len(combined_text)))
                text_mask = torch.tensor([1 if i < len(combined_text) else 0 for i in range(self.max_len)])
                post_ids = torch.tensor([ord(c) % 250000 for c in post_clean[:self.max_len]] + [0] * max(0, self.max_len - len(post_clean)))
                post_mask = torch.tensor([1 if i < len(post_clean) else 0 for i in range(self.max_len)])
            
            if len(self._token_cache) < 25000:
                self._token_cache[idx] = (text_ids, text_mask, post_ids, post_mask, has_post_bool)

        # 4. Binary & Multi-Label Targets
        binary_label = torch.tensor(int(item.get("label", 0)), dtype=torch.long)
        has_category_labels = torch.tensor(item.get("has_category_labels", True), dtype=torch.bool)
        has_post = torch.tensor(has_post_bool, dtype=torch.bool)
        
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
            "has_post": has_post,
            "has_image": torch.tensor(has_image_bool, dtype=torch.bool),
            "has_category_labels": has_category_labels,
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
