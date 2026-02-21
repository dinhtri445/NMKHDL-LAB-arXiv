# src/model_training.py
import os
import random
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from utils import load_json, save_json, parse_bibtex_entry
from feature_engineering import FeatureEngineer

log = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, result_root_dir):
        self.result_root = result_root_dir
        self.data_dir = os.path.join(self.result_root, "data_output")
        self.dataset_path = os.path.join(self.result_root, "dataset_ground_truth.json")
        self.features_path = os.path.join(self.result_root, "training_features.csv")
        
        self.manual_gt_path = os.path.join('src', 'manual_ground_truth.json')
        # Helper feature engineer để tính feature lúc predict
        self.fe = FeatureEngineer(result_root_dir)

    def split_papers(self, dataset):
        """
        Phân chia bài báo vào Train/Val/Test theo yêu cầu:
        - Test: 1 Manual + 1 Auto
        - Val:  1 Manual + 1 Auto
        - Train: Còn lại
        """
        manual_ids = []
        auto_ids = []
        
        # Phân loại dựa vào file manual config 
        manual_config_ids = set()
        if os.path.exists(self.manual_gt_path):
            manual_config_ids = set(load_json(self.manual_gt_path).keys())
            
        for pid in dataset.keys():
            if pid in manual_config_ids:
                manual_ids.append(pid)
            else:
                auto_ids.append(pid)
                
        # Shuffle để ngẫu nhiên
        random.seed(42)
        random.shuffle(manual_ids)
        random.shuffle(auto_ids)
        
        # Kiểm tra đủ số lượng tối thiểu
        if len(manual_ids) < 2:
            log.warning("Not enough manual papers for strict split (Need at least 2). Using available ones.")
        
        partitions = {"test": [], "val": [], "train": []}
        
        # 1. Chọn Test Set
        if manual_ids: partitions["test"].append(manual_ids.pop(0))
        if auto_ids:   partitions["test"].append(auto_ids.pop(0))
        
        # 2. Chọn Validation Set
        if manual_ids: partitions["val"].append(manual_ids.pop(0))
        if auto_ids:   partitions["val"].append(auto_ids.pop(0))
        
        # 3. Train Set (Phần còn lại)
        partitions["train"].extend(manual_ids)
        partitions["train"].extend(auto_ids)
        
        log.info(f"Data Split Summary:")
        log.info(f" - Test : {len(partitions['test'])} papers {partitions['test']}")
        log.info(f" - Val  : {len(partitions['val'])} papers {partitions['val']}")
        log.info(f" - Train: {len(partitions['train'])} papers")
        
        return partitions

    def train_model(self, partitions):
        """
        Huấn luyện mô hình Random Forest trên tập Train
        """
        if not os.path.exists(self.features_path):
            log.error("Training features not found!")
            return None

        # Load toàn bộ feature data
        df = pd.read_csv(self.features_path)
        
        # Lọc chỉ lấy các dòng thuộc tập Train
        train_pids = set(partitions['train'])
        train_df = df[df['paper_id'].isin(train_pids)]
        
        if len(train_df) == 0:
            log.error("Training set is empty!")
            return None

        # Chọn feature columns (bỏ các cột metadata)
        feature_cols = [c for c in df.columns if c not in ['label', 'paper_id', 'bib_key', 'candidate_id']]
        X_train = train_df[feature_cols]
        y_train = train_df['label']
        
        log.info(f"Training Model on {len(train_df)} samples with features: {feature_cols}")
        
        # Khởi tạo và Train model
        # Dùng Random Forest vì nó xử lý tốt feature không đồng nhất (số, tỉ lệ)
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        log.info("-" * 40)
        log.info("FEATURE IMPORTANCE ANALYSIS")
        log.info("-" * 40)
        
        importances = model.feature_importances_
        # Sắp xếp giảm dần
        indices = np.argsort(importances)[::-1]
        
        # In Top 3 (hoặc tất cả nếu ít hơn 3)
        top_k = 3
        for i in range(min(top_k, len(feature_cols))):
            idx = indices[i]
            feat_name = feature_cols[idx]
            score = importances[idx]
            log.info(f" {i+1}. {feat_name:<20}: {score:.4f}")
            
        log.info("-" * 40)
        log.info("Model training completed.")
        return model, feature_cols

    def predict_paper(self, paper_id, model, feature_cols, partition_name, ground_truth_map):
        """
        Quy trình dự đoán Full cho 1 bài báo:
        1. Tạo cặp (Bib x All Candidates) -> Feature Extraction
        2. Predict Proba
        3. Ranking Top 5
        4. Tính MRR
        5. Lưu pred.json
        """
        bib_path = os.path.join(self.data_dir, paper_id, 'refs.bib')
        ref_json_path = os.path.join(self.data_dir, paper_id, 'references.json')
        
        if not os.path.exists(bib_path) or not os.path.exists(ref_json_path):
            return None, 0.0, 0.0

        # Load data
        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_entries = parse_bibtex_entry(f.read())
        json_refs = load_json(ref_json_path)
        candidate_ids = list(json_refs.keys())
        
        if not candidate_ids: return None, 0.0, 0.0

        prediction_output = {}
        mrr_sum = 0
        mrr_count = 0
        
        # Duyệt qua từng Bib Entry
        for entry in bib_entries:
            bib_key = entry['ID']
            
            # --- BƯỚC 1: TẠO FEATURE CHO TẤT CẢ CANDIDATES ---
            # Để xếp hạng, ta phải so sánh bib_entry này với TẤT CẢ candidate có trong json
            # (Thay vì chỉ sample 5 cái như lúc train)
            features_list = []
            cand_list = []
            
            for cid in candidate_ids:
                feats = self.fe.extract_features(entry, json_refs[cid])
                # Đảm bảo thứ tự cột đúng như lúc train
                feat_vector = [feats.get(col, 0) for col in feature_cols]
                features_list.append(feat_vector)
                cand_list.append(cid)
            
            if not features_list:
                prediction_output[bib_key] = []
                continue

            # --- BƯỚC 2: PREDICT SCORE ---
            # Dự đoán xác suất nhãn 1 (Match)
            X_pred = pd.DataFrame(features_list, columns=feature_cols)
            probs = model.predict_proba(X_pred)[:, 1] # Lấy cột xác suất class 1
            
            # --- BƯỚC 3: RANKING ---
            # Ghép candidate id với score và sort giảm dần
            scored_candidates = list(zip(cand_list, probs))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy Top 5 IDs
            top_5 = [x[0] for x in scored_candidates[:5]]
            prediction_output[bib_key] = top_5
            
            # --- BƯỚC 4: TÍNH MRR (Chỉ tính nếu có Ground Truth dương) ---
            true_id = ground_truth_map.get(bib_key)
            if true_id: # Nếu Ground truth không phải null
                if true_id in top_5:
                    rank = top_5.index(true_id) + 1 # Rank bắt đầu từ 1
                    mrr_sum += 1.0 / rank
                else:
                    mrr_sum += 0.0 # Không tìm thấy trong top 5
                mrr_count += 1
                
        # --- BƯỚC 5: LƯU KẾT QUẢ ---
        final_json = {
            "partition": partition_name,
            "groundtruth": ground_truth_map,
            "prediction": prediction_output
        }
        
        pred_path = os.path.join(self.data_dir, paper_id, 'pred.json')
        save_json(final_json, pred_path)
        
        return final_json, mrr_sum, mrr_count

    def run_evaluation(self):
        """
        Chạy toàn bộ quy trình Train -> Eval -> Output
        """
        # 1. Load Dataset
        if not os.path.exists(self.dataset_path):
            log.error("Dataset not found!")
            return {}
        dataset = load_json(self.dataset_path)
        
        # 2. Split Data
        partitions = self.split_papers(dataset)
        
        # 3. Train Model
        model, feat_cols = self.train_model(partitions)
        if not model: return
        
        # 4. Evaluate & Generate pred.json (Cho tất cả partitions)
        # Nhưng chỉ tính thống kê MRR cho TEST SET
        total_mrr_sum = 0
        total_mrr_count = 0
        
        # Thứ tự chạy: Train -> Val -> Test (để file log nhìn logic)
        for part_name in ['train', 'val', 'test']:
            pids = partitions[part_name]
            if not pids: continue
            
            log.info(f"Generating predictions for {part_name} set ({len(pids)} papers)...")
            
            for pid in tqdm(pids, desc=f"Processing {part_name}"):
                gt_map = dataset.get(pid, {})
                _, mrr_sum, mrr_count = self.predict_paper(pid, model, feat_cols, part_name, gt_map)
                
                # Chỉ lưu MRR nếu là tập Test
                if part_name == 'test':
                    total_mrr_sum += mrr_sum
                    total_mrr_count += mrr_count    
        
        # 4. Final Stats
        avg_test_mrr = total_mrr_sum / total_mrr_count if total_mrr_count > 0 else 0.0
        
        log.info("="*30)
        log.info("MODEL EVALUATION RESULTS")
        log.info("="*30)
        log.info(f"Test Set MRR : {avg_test_mrr:.4f}")
        log.info("="*30)
        
        return avg_test_mrr
        