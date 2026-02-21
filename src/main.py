# src/main.py
import argparse
import os
import json
import logging
from parser_pipeline import LatexParser
from matching_pipeline import MatchingPipeline
from logging_config import setup_logger
from utils import save_json

log = logging.getLogger(__name__)

def run_task_parsing(input_dir, data_dir, report_path):
    """
    Xử lý Task 2.1: Parsing
    """
    papers = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Init Report
    report = {
        "summary": {
            "total_papers": len(papers),
            "processed_papers": 0,
            "papers_no_tex": 0,
            "papers_with_hierarchy": 0,
            "papers_with_refs": 0,
            "avg_hierarchy_depth": 0,
            "avg_max_depth": 0,
            "avg_branching_factor": 0,
            "avg_refs_count": 0
        },
    }

    # Biến tính trung bình
    sum_depth = 0; sum_max_depth = 0; sum_branching = 0; sum_refs = 0
    count_hier = 0; count_refs = 0

    log.info(f"Running Parsing Task on {len(papers)} papers...")
    log.info(f"Data Output: {data_dir}")

    for paper_id in papers:
        log.info(f"=== Processing {paper_id} ===")
        paper_path = os.path.join(input_dir, paper_id)
        
        try:
            # Truyền data_dir (thư mục con chứa data) vào Parser
            parser_obj = LatexParser(paper_path, data_dir)
            stats = parser_obj.run_parsing()
            
            # Update Report
            report["summary"]["processed_papers"] += 1
            if not stats["has_tex"]: report["summary"]["papers_no_tex"] += 1
            
            if stats["hierarchy_success"]:
                report["summary"]["papers_with_hierarchy"] += 1
                count_hier += 1
                sum_depth += stats["avg_depth"]
                sum_max_depth += stats["max_depth"]
                sum_branching += stats["avg_branching"]
            
            if stats["bib_created"]:
                report["summary"]["papers_with_refs"] += 1
                count_refs += 1
                sum_refs += stats["refs_count"]

        except Exception as e:
            log.error(f"CRASH processing {paper_id}: {e}", exc_info=True)

    # Tính trung bình
    if count_hier > 0:
        report["summary"]["avg_hierarchy_depth"] = round(sum_depth / count_hier, 2)
        report["summary"]["avg_max_depth"] = round(sum_max_depth / count_hier, 2)
        report["summary"]["avg_branching_factor"] = round(sum_branching / count_hier, 2)
    
    if count_refs > 0:
        report["summary"]["avg_refs_count"] = round(sum_refs / count_refs, 2)

    # Lưu Report vào thư mục stats
    save_json(report, report_path)
    
    log.info("="*30)
    log.info("TASK 2.1 FINISHED. SUMMARY:")
    log.info(json.dumps(report["summary"], indent=4))
    log.info(f"Statistics saved to: {report_path}")
    log.info(f"Parsing Report saved to {report_path}")
    log.info("="*30)

def run_task_matching(result_dir, report_path):
    """
    Xử lý Task 2.2: Reference Matching Pipeline
    Input: result_dir (Thư mục gốc chứa data_output từ task 2.1)
    Output: result_dir (Thêm các file pred.json và dataset_ground_truth.json)
    """
    log.info("="*30)
    log.info("===== Reference Matching Pipeline =====")
    log.info(f"Working Directory: {result_dir}")
    log.info("="*30)
    
    try:
        # Khởi tạo Pipeline với thư mục gốc result
        matcher = MatchingPipeline(result_dir)
        
        # Chạy pipeline
        result_stats = matcher.run_matching()
        
        # Lưu report thống kê
        save_json(result_stats, report_path)
        log.info(f"Matching Task Completed. Stats saved to {report_path}")
        
    except Exception as e:
        log.error(f"Matching Pipeline Failed: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Lab Pipeline Manager")
    parser.add_argument('--input_dir', type=str, default='data', help='Folder chứa data')
    # Tham số này giờ sẽ là thư mục GỐC chứa tất cả kết quả (result)
    parser.add_argument('--result_dir', type=str, default='result', help='Folder chứa toàn bộ kết quả (logs, stats, data)')
    
    parser.add_argument('--task', type=str, default='parsing', choices=['parsing', 'matching'], 
                        help='Chọn task: parsing (2.1) hoặc matching (2.2)')
    
    args = parser.parse_args()

    # 1. Kiểm tra Input
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found!")
        return

    
    dir_data = os.path.join(args.result_dir, "data_output")
    dir_stats = os.path.join(args.result_dir, "stats")
    dir_logs = os.path.join(args.result_dir, "logs")

    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_stats, exist_ok=True)
    os.makedirs(dir_logs, exist_ok=True)

    # 3. Điều hướng Task & Cấu hình Log riêng biệt
    if args.task == 'parsing':
        # Cấu hình log cho parsing
        log_file = os.path.join(dir_logs, "parsing.log")
        setup_logger(log_file)
        
        # Đường dẫn file report
        report_file = os.path.join(dir_stats, "parsing_report.json")
        
        # Chạy task
        run_task_parsing(args.input_dir, dir_data, report_file)

    elif args.task == 'matching':
        # Cấu hình log cho matching
        log_file = os.path.join(dir_logs, "matching.log")
        setup_logger(log_file)
        
        # Đường dẫn file report
        report_file = os.path.join(dir_stats, "matching_report.json")
        
        # Chạy task
        # input_dir là data_output có được từ parse
        run_task_matching(args.input_dir, report_file)

if __name__ == "__main__":
    main()