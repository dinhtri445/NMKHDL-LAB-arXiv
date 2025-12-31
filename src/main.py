
import argparse
import sys
import os
import logging
from datetime import datetime
import concurrent.futures
import threading

# Import các file tự xây dựng
from config import Config
from metadata_harvester import MetadataHarvester
from source_downloader import SourceDownloader
from reference_extractor import ReferenceExtractor
from logger import setup_logging

class ArXivScraper:
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger('main')
        
        self.metadata_harvester = MetadataHarvester()
        self.source_downloader = SourceDownloader()
        self.reference_extractor = ReferenceExtractor()
        
        self.lock = threading.Lock()  # <-- lock cho cập nhật đồng bộ
        self.stats = {
            'start_time': None,
            'total_time': None,
            'total_papers': 0,
            'successful_papers': 0,
            'failed_papers': 0,
            'total_versions': 0,
            'total_references': 0,
            'reference_success_count':0,
            'memory_usage': [],
            'disk_usage': 0,
            'current_index': 0,
            'total_original_size': 0,
            'total_extracted_size': 0,
            'processed_papers': set()
        }

    def generate_arxiv_ids(self):
        """Generate arXiv IDs trong phạm vi được giao"""
        base_ids = []
        for i in range(Config.START_ID, Config.END_ID + 1):
            base_id = f"{Config.YYMM}.{str(i).zfill(5)}"  
            base_ids.append(base_id)
        return base_ids

    def load_progress(self):
        state = Config.load_state()
        if state:
            self.stats.update(state)
            if 'processed_papers' in state:
                self.stats['processed_papers'] = set(state['processed_papers'])
            self.logger.info(f"Loaded progress: {self.stats.get('current_index',0)}/{self.stats.get('total_papers',0)}")
            return True
        return False

    def save_progress(self):
        state = {
            'current_index': self.stats['current_index'],
            'total_time': self.stats['total_time'],
            'successful_papers': self.stats['successful_papers'],
            'overall success rate': (self.stats['successful_papers'] / self.stats['current_index']) if self.stats['current_index'] else 0,
            'failed_papers': self.stats['failed_papers'],
            'total_versions': self.stats['total_versions'],
            'total_references': self.stats['total_references'],
            'reference_success_rate': (self.stats['reference_success_count'] / self.stats['successful_papers']) if self.stats['successful_papers'] else 0,
            'processed_papers': list(self.stats['processed_papers']),
            'avg_refs' : (self.stats['total_references'] / self.stats['successful_papers']) if self.stats['successful_papers'] else 0,
            'avg_memory' : sum(self.stats['memory_usage']) / len(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
            'max_memory' : max(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
            'total_original_size': self.stats['total_original_size'],
            'total_extracted_size': self.stats['total_extracted_size'],
            'avg_original' : (self.stats['total_original_size'] / self.stats['successful_papers']) if self.stats['successful_papers'] else 0,
            'avg_extracted' : (self.stats['total_extracted_size'] / self.stats['successful_papers']) if self.stats['successful_papers'] else 0,
            'disk_usage' : self.stats['disk_usage'],
            'last_save': datetime.now().isoformat()
        }
        Config.save_state(state)
        self.logger.info(f"Saved progress: {self.stats['current_index']}")

    def process_single_paper(self, base_id):
        """
        Process a single paper. Returns a dict with result info:
        {
        'base_id': base_id,
        'success': True/False,
        'versions': int,
        'references': int,
        'error': optional error message
        }
        """
        try:
            with self.lock:
                if base_id in self.stats['processed_papers']:
                    self.logger.info(f"Skipping already processed {base_id}")
                    return {'base_id': base_id, 'success': True, 'versions': 0, 'references': 0}

            paper_folder = base_id.replace('.', '-')
            self.logger.info(f"Processing {base_id}")

            # 1) Metadata
            metadata = self.metadata_harvester.fetch_single_paper_metadata(base_id)
            if not metadata:
                self.logger.warning(f"No metadata versions found for {base_id}")
                return {'base_id': base_id, 'success': False, 'versions': 0, 'references': 0, 'error': 'no metadata'}

            # Save metadata
            self.metadata_harvester.save_metadata_json(metadata, paper_folder)

            
            # 2) References
            references = self.reference_extractor.extract_references(base_id) or {}
            if references:
                self.reference_extractor.save_references_json(references, paper_folder)

            # 3) Download sources
            versions_downloaded, original_size, extracted_size = self.source_downloader.download_all_versions(base_id) or ([], 0, 0)
            if not versions_downloaded:
                self.logger.warning(f"No sources downloaded for {base_id}")
                
            
            self.stats['total_original_size'] += original_size
            self.stats['total_extracted_size'] += extracted_size

        
            return {
                'base_id': base_id,
                'success': True,
                'versions': len(versions_downloaded),
                'references': len(references)
            }

        except Exception as e:
            self.logger.error(f"Error processing {base_id}: {e}")
            return {'base_id': base_id, 'success': False, 'versions': 0, 'references': 0, 'error': str(e)}

    def run_parallel(self, base_ids, workers: int = 2, save_interval: int = 10):
        """
        Xử lý base_ids bằng ThreadPoolExecutor với các luồng `workers`
        """
        total = len(base_ids)
        self.stats['total_papers'] = total
        self.logger.info(f"Total papers to process in this run: {total}")

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            
            futures = {executor.submit(self.process_single_paper, bid): bid for bid in base_ids}

            for fut in concurrent.futures.as_completed(futures):
                bid = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    self.logger.error(f"Unhandled exception for {bid}: {e}")
                    res = {'base_id': bid, 'success': False, 'versions': 0, 'references': 0, 'error': str(e)}

                # cập nhật số liệu thống kê theo cách an toàn cho luồng
                with self.lock:
                    completed += 1
                    self.stats['current_index'] += 1  
                    if res.get('success'):
                        self.stats['successful_papers'] += 1
                    else:
                        self.stats['failed_papers'] += 1
                    self.stats['total_versions'] += int(res.get('versions', 0))
                    self.stats['total_references'] += int(res.get('references', 0))
                    self.stats['processed_papers'].add(res['base_id'])
                    if res.get('references', 0) > 0:
                        self.stats['reference_success_count'] = self.stats.get('reference_success_count', 0) + 1
                    
                    try:
                        self.stats['memory_usage'].append(Config.get_memory_usage())
                    except Exception:
                        pass

                
                if completed % save_interval == 0:
                    self.save_progress()

                self.logger.info(f"--- Completed {completed}/{total}: {res['base_id']} success={res.get('success')} versions={res.get('versions')} refs={res.get('references')}")

        output_size = self.get_folder_size(Config.BASE_DIR)
        self.stats['disk_usage'] = max(self.stats.get('disk_usage', 0), output_size)
        
        self._print_final_stats()
        # final save
        self.save_progress()
        return (self.stats['successful_papers'], self.stats['failed_papers'])


    def run(self, start_index=0, batch_size=None, resume=False, save_interval=10, workers: int = 1):
        self.stats['start_time'] = datetime.now()
        if resume and self.load_progress():
            start_index = self.stats['current_index']
            self.logger.info(f"Resuming from index {start_index}")
        else:
            self.stats['current_index'] = start_index

        all_base_ids = self.generate_arxiv_ids()
        if batch_size:
            slice_ids = all_base_ids[start_index:start_index + batch_size]
        else:
            slice_ids = all_base_ids[start_index:]

        # Tùy chọn chạy song song hoặc tuần tự
        if workers and workers > 1:
            return self.run_parallel(slice_ids, workers=workers, save_interval=save_interval)
        else:            
            total = len(slice_ids)
            self.stats['total_papers'] = total
            self.logger.info(f"Total papers to process in this run: {total}")
            for i, base_id in enumerate(slice_ids, 1):
                self.logger.info(f"--- Processing {i}/{total}: {base_id} ---")
                self.stats['memory_usage'].append(Config.get_memory_usage())
                res = self.process_single_paper(base_id)
                
                if res.get('success'):
                    self.stats['successful_papers'] += 1
                    self.stats['total_versions'] += int(res.get('versions', 0))
                    self.stats['total_references'] += int(res.get('references', 0))
                    self.stats['processed_papers'].add(base_id)
                else:
                    self.stats['failed_papers'] += 1

                self.stats['current_index'] = start_index + i
                if i % save_interval == 0:
                    self.save_progress()

            output_size = self.get_folder_size(Config.BASE_DIR)
            self.stats['disk_usage'] = max(self.stats.get('disk_usage', 0), output_size)

            
            self._print_final_stats()
            self.save_progress()
            return (self.stats['successful_papers'], self.stats['failed_papers'])


    def _print_final_stats(self):
        end_time = datetime.now()
        total_time = end_time - self.stats['start_time']
        avg_refs = self.stats['total_references'] / self.stats['successful_papers'] if self.stats['successful_papers'] else 0
        avg_memory = sum(self.stats['memory_usage']) / len(self.stats['memory_usage']) if self.stats['memory_usage'] else 0
        max_memory = max(self.stats['memory_usage']) if self.stats['memory_usage'] else 0
        avg_time_per_paper = total_time / self.stats['current_index'] if self.stats['current_index'] else 0

        self.stats['total_time'] = total_time.total_seconds()

        self.logger.info("SCRAPING COMPLETED")
        self.logger.info(f"Total processed: {self.stats['current_index']}")
        self.logger.info(f"Successful papers: {self.stats['successful_papers']}")
        self.logger.info(f"Failed papers: {self.stats['failed_papers']}")
        self.logger.info(f"Total versions: {self.stats['total_versions']}")
        self.logger.info(f"Total references: {self.stats['total_references']}")
        self.logger.info(f"Average references per paper: {avg_refs:.2f}")
        self.logger.info(f"Average time per paper (s): {avg_time_per_paper}")
        self.logger.info(f"Max memory usage: {max_memory:.2f}")
        self.logger.info(f"Average memory usage: {avg_memory:.2f}")
        self.logger.info(f"Total scraping time (s): {total_time}")
        self.logger.info(f"Total original size': {self.stats['total_original_size']}")
        self.logger.info(f"Total extracted size': {self.stats['total_extracted_size']}")

        

    def get_folder_size(self, folder):
        total = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total
        

def main():
    parser = argparse.ArgumentParser(description='arXiv Scraper for Lab 1')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--student_id', type=str, default="23120377", help='Student ID')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--save_interval', type=int, default=10, help='Papers between progress saves')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (threads) to use')
    args = parser.parse_args()

    Config.update_student_id(args.student_id)

    print("ArXiv Scraper - Lab 1: Data Science")
    print(f"Student ID: {Config.STUDENT_ID}")
    print(f"Target range: {Config.YYMM}.{Config.START_ID} to {Config.YYMM}.{Config.END_ID}")
    print("="*50)

    scraper = ArXivScraper()
    try:
        successful, failed = scraper.run(
            start_index=args.start, 
            batch_size=args.batch, 
            resume=args.resume, 
            workers=args.workers,
            save_interval=args.save_interval
        )
        if successful > 0:
            print(f"Successfully completed: {successful} papers.")
        else:
            print("No papers were successfully processed.")
        return successful > 0
    except KeyboardInterrupt:
        print("Process interrupted by user. Progress saved.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)