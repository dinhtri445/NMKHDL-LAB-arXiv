# src/feature_engineering.py
import os
import random
import logging
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import load_json, parse_bibtex_entry
from cleaning import ReferenceCleaner

log = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, result_root_dir):
        self.result_root = result_root_dir
        self.data_dir = os.path.join(self.result_root, "data_output")
        self.dataset_path = os.path.join(self.result_root, "dataset_ground_truth.json")
        self.output_features_path = os.path.join(self.result_root, "training_features.csv")
        
        self.cleaner = ReferenceCleaner()
        

        self.negative_sample_ratio = 5 
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

    # ==========================================
    # 1. FEATURE CALCULATION FUNCTIONS
    # ==========================================
    def _compute_string_similarity(self, s1, s2):
        """Tính độ tương đồng chuỗi dùng SequenceMatcher (tương tự Levenshtein)"""
        if not s1 or not s2: return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    def _compute_jaccard_similarity(self, set1, set2):
        """Tính Jaccard: (Giao) / (Hợp)"""
        if not set1 or not set2: return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _compute_tfidf_cosine(self, s1, s2):
        try:
            # Tạo corpus nhỏ chỉ gồm 2 câu này để tính nhanh
            # (Lưu ý: Cách này không chuẩn TF-IDF toàn cục nhưng đủ dùng cho so sánh cặp)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([s1, s2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0
        
    def extract_features(self, bib_entry, json_ref):
        """
        Trích xuất vector đặc trưng cho 1 cặp (BibTeX, JSON Ref)
        """
        features = {}

        # --- PREPARE DATA ---
        # Title
        t1 = self.cleaner.clean_title(bib_entry.get('title', ''))
        t2 = self.cleaner.clean_title(json_ref.get('paper_title', ''))
        
        # Author (Set of normalized names)
        a1 = self.cleaner.clean_authors(bib_entry.get('author', ''))
        a2 = self.cleaner.clean_authors(json_ref.get('authors', []))
        
        # Year
        y1 = self.cleaner.extract_year(bib_entry.get('year'))
        y2 = self.cleaner.extract_year(json_ref.get('submission_date'))

        # --- 1. TITLE FEATURES ---
        features['title_sim_score'] = self._compute_string_similarity(t1, t2)
        
        # Token Jaccard cho Title
        t1_tokens = set(t1.split())
        t2_tokens = set(t2.split())
        features['title_jaccard'] = self._compute_jaccard_similarity(t1_tokens, t2_tokens)
        
        features['title_len_diff'] = abs(len(t1) - len(t2))

        features['title_tfidf_cosine'] = self._compute_tfidf_cosine(t1, t2)

        # --- 2. AUTHOR FEATURES ---
        features['author_jaccard'] = self._compute_jaccard_similarity(a1, a2)
        features['common_authors'] = len(a1.intersection(a2))

        # --- 3. YEAR FEATURES ---
        if y1 is not None and y2 is not None:
            diff = abs(y1 - y2)
            features['year_diff'] = diff
            features['year_match'] = 1 if diff <= 1 else 0
        else:
            # Xử lý Missing Value: Gán giá trị mặc định "xấu"
            features['year_diff'] = 10 
            features['year_match'] = 0

        return features

    # ==========================================
    # 2. DATASET GENERATION
    # ==========================================
    def create_training_dataset(self):
        """
        Tạo file CSV chứa Features và Label từ Ground Truth Dataset.
        """
        log.info("--- Starting Feature Engineering Phase ---")
        
        # Load Ground Truth
        if not os.path.exists(self.dataset_path):
            log.error("Ground truth dataset not found. Please run Labelling step first.")
            return None
        
        ground_truth = load_json(self.dataset_path)
        log.info(f"Loaded ground truth for {len(ground_truth)} papers.")

        dataset_rows = []
        
        # Duyệt qua từng bài báo trong tập Ground Truth
        for pid, mapping in tqdm(ground_truth.items(), desc="Extracting Features"):
            
            # Load Raw Data của bài báo đó
            bib_path = os.path.join(self.data_dir, pid, 'refs.bib')
            ref_json_path = os.path.join(self.data_dir, pid, 'references.json')
            
            if not os.path.exists(bib_path) or not os.path.exists(ref_json_path):
                continue

            try:
                with open(bib_path, 'r', encoding='utf-8') as f:
                    bib_entries = parse_bibtex_entry(f.read())
                json_refs = load_json(ref_json_path)
                
                # Convert list bib entries to dict for fast access
                bib_dict = {e['ID']: e for e in bib_entries}
                candidate_ids = list(json_refs.keys()) # Danh sách tất cả ArXiv ID có thể match
                
                # Duyệt qua từng nhãn đã gán (BibKey -> TargetID)
                for bib_key, target_id in mapping.items():
                    if bib_key not in bib_dict: continue
                    
                    bib_entry = bib_dict[bib_key]
                    
                    # --- A. POSITIVE SAMPLE (Nếu target_id không phải null) ---
                    if target_id and target_id in json_refs:
                        pos_ref = json_refs[target_id]
                        feats = self.extract_features(bib_entry, pos_ref)
                        feats['label'] = 1
                        feats['paper_id'] = pid
                        feats['bib_key'] = bib_key
                        feats['candidate_id'] = target_id
                        dataset_rows.append(feats)

                    # --- B. NEGATIVE SAMPLES ---
                    # Lấy ngẫu nhiên k candidates KHÔNG PHẢI là target_id
                    neg_candidates = [cid for cid in candidate_ids if cid != target_id]
                    
                    # Nếu số lượng candidate ít, lấy hết. Nếu nhiều, sample bớt.
                    sample_size = min(len(neg_candidates), self.negative_sample_ratio)
                    selected_negs = random.sample(neg_candidates, sample_size)
                    
                    for neg_id in selected_negs:
                        neg_ref = json_refs[neg_id]
                        feats = self.extract_features(bib_entry, neg_ref)
                        feats['label'] = 0
                        feats['paper_id'] = pid
                        feats['bib_key'] = bib_key
                        feats['candidate_id'] = neg_id
                        dataset_rows.append(feats)
                        
            except Exception as e:
                log.warning(f"Error processing paper {pid}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(dataset_rows)
        
        # Save to CSV
        df.to_csv(self.output_features_path, index=False)
        log.info(f"Feature Engineering Completed.")
        log.info(f"Generated {len(df)} samples (Rows).")
        log.info(f"Saved training data to: {self.output_features_path}")
        
        return df