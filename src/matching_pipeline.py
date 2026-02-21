# src/matching_pipeline.py
import os
import logging
import numpy as np # Cần thêm thư viện này để tính trung bình nhanh
from tqdm import tqdm
from utils import load_json, save_json, parse_bibtex_entry
from cleaning import ReferenceCleaner
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

log = logging.getLogger(__name__)

class MatchingPipeline:
    def __init__(self, result_root_dir):
        """
        Khởi tạo Pipeline cho Task 2.2
        :param result_root_dir: Thư mục gốc chứa kết quả (VD: result/)
        """
        self.result_root = result_root_dir
        self.data_dir = os.path.join(self.result_root, "data_output")
        self.final_dataset_path = os.path.join(self.result_root, 'dataset_ground_truth.json')
        # Trỏ vào file manual trong thư mục src
        self.manual_gt_path = os.path.join('src', 'manual_ground_truth.json')
        self.cleaner = ReferenceCleaner()

    def _match_heuristic(self, bib_entry, json_refs):
        """Logic so khớp tự động (Silver Standard)"""
        bib_title = self.cleaner.clean_title(bib_entry.get('title', ''))
        bib_year = self.cleaner.extract_year(bib_entry.get('year'))
        if not bib_title: return None

        for ref_id, ref_data in json_refs.items():
            json_title = self.cleaner.clean_title(ref_data.get('paper_title', ''))
            json_year = self.cleaner.extract_year(ref_data.get('submission_date'))
            
            if bib_title == json_title:
                if bib_year and json_year:
                    if abs(bib_year - json_year) <= 1: return ref_id
                elif len(bib_title) > 20: return ref_id
        return None

    def prepare_ground_truth_dataset(self, target_total=500):
        """
        Tạo dataset với log chi tiết và thống kê.
        """
        log.info(f"{'='*10} STARTING DATA LABELLING PHASE (Target: {target_total}) {'='*10}")
        
        final_dataset = {} 
        manual_ids = set()

        # ==========================================
        # PHẦN 1: LOAD MANUAL DATA (VỚI LOG ĐẸP)
        # ==========================================
        if os.path.exists(self.manual_gt_path):
            try:
                manual_data = load_json(self.manual_gt_path)
                if manual_data:
                    log.info(f"Loading Manual Ground Truth from: {self.manual_gt_path}")
                    log.info("-" * 65)
                    log.info(f"{'Paper ID':<15} | {'Total':<8} | {'Matched':<8} | {'Null':<8} | {'Status'}")
                    log.info("-" * 65)

                    for pid, mappings in manual_data.items():
                        total = len(mappings)
                        matched = sum(1 for v in mappings.values() if v is not None)
                        nulls = total - matched
                        final_dataset[pid] = mappings
                        manual_ids.add(pid)
                        log.info(f"{pid:<15} | {total:<8} | {matched:<8} | {nulls:<8} | {'Loaded'}")
                    
                    log.info("-" * 65)
                    log.info(f"Successfully loaded {len(manual_ids)} manual papers.")
            except Exception as e:
                log.error(f"Error loading manual file: {e}")
        else:
            log.warning(f"Manual file not found at {self.manual_gt_path}")

        # ==========================================
        # PHẦN 2: AUTO LABELLING (VỚI THỐNG KÊ)
        # ==========================================
        target_auto = max(0, target_total - len(manual_ids))
        if target_auto == 0:
            log.info("Target reached with manual data. Skipping auto-labeling.")
            save_json(final_dataset, self.final_dataset_path)
            return final_dataset

        # Scan Candidates
        if not os.path.exists(self.data_dir):
            log.error("Data output directory missing!")
            return {}

        all_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        candidates = []
        
        # Lọc sơ bộ
        log.info("Scanning candidates...")
        for pid in all_dirs:
            if pid in manual_ids: continue
            bib_path = os.path.join(self.data_dir, pid, 'refs.bib')
            ref_json_path = os.path.join(self.data_dir, pid, 'references.json')
            if os.path.exists(bib_path) and os.path.exists(ref_json_path):
                candidates.append(pid)
        
        log.info(f"Found {len(candidates)} valid candidates for auto-labeling.")
        
        # Thống kê cho quá trình chạy
        stats = {
            "scanned": 0,
            "accepted": 0,
            "rejected_small_size": 0,
            "rejected_low_match": 0,
            "rejected_error": 0,
            "total_match_rate": []
        }

        # Thanh tiến trình hiển thị tiến độ hoàn thành MỤC TIÊU
        pbar = tqdm(total=target_auto, desc="Auto Labelling Progress", unit="paper")
        
        for pid in candidates:
            if stats["accepted"] >= target_auto: break
            
            stats["scanned"] += 1
            
            try:
                bib_path = os.path.join(self.data_dir, pid, 'refs.bib')
                ref_json_path = os.path.join(self.data_dir, pid, 'references.json')
                
                with open(bib_path, 'r', encoding='utf-8') as f:
                    bib_entries = parse_bibtex_entry(f.read())
                json_refs = load_json(ref_json_path)
                
                if not json_refs or not bib_entries:
                    stats["rejected_error"] += 1
                    continue
                
                # Xét bài báo có ít nhất 20 cặp tham chiếu
                total_refs = len(bib_entries)
                if total_refs < 20:
                    stats["rejected_small_size"] += 1
                    continue 

                paper_mapping = {}
                match_hits = 0
                
                for entry in bib_entries:
                    match_id = self._match_heuristic(entry, json_refs)
                    if match_id:
                        paper_mapping[entry['ID']] = match_id
                        match_hits += 1
                
                # Tính tỷ lệ match
                total_refs = len(bib_entries)
                match_rate = match_hits / total_refs if total_refs > 0 else 0
                
                # ĐIỀU KIỆN CHẤP NHẬN: > 30% match
                if total_refs > 0 and match_rate > 0.3:
                    final_dataset[pid] = paper_mapping
                    stats["accepted"] += 1
                    stats["total_match_rate"].append(match_rate)
                    pbar.update(1)
                    
                else:
                    stats["rejected_low_match"] += 1
                    
            except Exception as e:
                stats["rejected_error"] += 1

        pbar.close()

        # ==========================================
        # PHẦN 3: TỔNG KẾT (REPORT)
        # ==========================================
        save_json(final_dataset, self.final_dataset_path)
        
        avg_rate = np.mean(stats["total_match_rate"]) if stats["total_match_rate"] else 0
        
        log.info("\n" + "="*30)
        log.info(" LABELLING STATISTICS REPORT")
        log.info("="*30)
        log.info(f"Total Papers in Dataset: {len(final_dataset)}")
        log.info(f"  - Manual:            {len(manual_ids)}")
        log.info(f"  - Auto-generated:    {stats['accepted']}")
        log.info("-" * 30)
        log.info(f"Candidates Scanned:    {stats['scanned']}")
        log.info(f"Rejected (<20 refs):    {stats['rejected_small_size']} (Too small)")
        log.info(f"Rejected (Low Match):  {stats['rejected_low_match']}")
        log.info(f"Rejected (Errors):     {stats['rejected_error']}")
        log.info("-" * 30)
        log.info(f"Avg Match Rate (Auto): {avg_rate:.1%} (Quality Indicator)")
        log.info("="*30)
        log.info(f"Dataset saved to: {self.final_dataset_path}")
        
        return final_dataset
    
    def run_matching(self):
        """
        Điều phối toàn bộ quy trình Task 2.2
        """
        # # Bước 1: Tạo Dataset (Labelling)
        # dataset = self.prepare_ground_truth_dataset(target_total=500)
        
        # # Bước 2: Feature Engineering
        # log.info("Starting Step 2: Feature Engineering...")
        # fe = FeatureEngineer(self.result_root)
        # df_features = fe.create_training_dataset()
        
        # [MỚI] Bước 3 & 4 & 5: Data Modeling, Eval, Prediction
        log.info("Starting Step 3: Model Training & Evaluation...")
        trainer = ModelTrainer(self.result_root)
        eval_results = trainer.run_evaluation()
        
        return {
            "status": "Matching Pipeline Completed", 
            #"dataset_size": len(dataset),
           # "features_generated": len(df_features) if df_features is not None else 0,
            "evaluation": eval_results
        }