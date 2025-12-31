import arxiv
import time
import logging
from typing import Dict, Any
from config import Config
import os
import json

logger = logging.getLogger('Metadata Harvester')

class MetadataHarvester:
    def __init__(self, per_result_delay: float = 1.2):
        """
        per_result_delay: số giây tạm nghỉ giữa các request để tuân thủ arXiv API.
        """
        self.client = arxiv.Client()
        self.per_result_delay = per_result_delay
        self.logger = logger

    def fetch_single_paper_metadata(self, base_id: str) -> Dict[str, Any]:
        """
        Lấy metadata của một paper duy nhất dựa vào base_id (ví dụ "2406.4361").
        Trả về dictionary format sẵn sàng lưu metadata.json
        """
        versions = []
        # Biến này sẽ lưu thông tin journal_ref tốt nhất tìm được
        final_journal_ref = None
        for v in range(1, Config.MAX_VERSIONS + 1):
            arxiv_id = f"{base_id}v{v}"
            try:
                # Tuân thủ rate limit
                time.sleep(self.per_result_delay)
                
                # Tìm paper theo arXiv ID
                search = arxiv.Search(id_list=[arxiv_id], max_results=1)
                it = self.client.results(search)
                entry = next(it, None)
                if entry is None:
                    break  # không còn version nào
                
                if entry.journal_ref:
                    final_journal_ref = entry.journal_ref

                # Lấy metadata cho version này
                meta = {
                    "version": v,
                    "arxiv_id": arxiv_id,
                    "title": getattr(entry, "title", ""),
                    "authors": [a.name for a in getattr(entry, "authors", [])] if getattr(entry, "authors", None) else [],
                    "summary": getattr(entry, "summary", ""),
                    "published": getattr(entry, "published", None).isoformat() if getattr(entry, "published", None) else "",
                    "updated": getattr(entry, "updated", None).isoformat() if getattr(entry, "updated", None) else "",
                    "entry_id": getattr(entry, "entry_id", "")
                }
                versions.append(meta)
            except Exception as e:
                self.logger.warning(f"Error fetching metadata for {arxiv_id}: {e}")
                break

        if not versions:
            self.logger.warning(f"No metadata found for {base_id}")
            return {}

        # Kết hợp thông tin tất cả version thành 1 dict theo format metadata.json
        combined_metadata = {
            "paper_title": versions[-1].get("title", ""),
            "authors": versions[-1].get("authors", []),    
            "submission_date": versions[0].get("published", ""),
            "revised_dates": [v.get("updated", "") for v in versions if v.get("updated")],
            "publication_venue": final_journal_ref
        }
        return combined_metadata

    def save_metadata_json(self, metadata: Dict[str, Any], paper_folder: str = "", filename: str = "metadata.json") -> bool:
        """
        Lưu metadata dictionary ra file JSON trong thư mục BASE_DIR.
        """
        try:
            os.makedirs(os.path.join(Config.BASE_DIR, paper_folder), exist_ok=True)
            filepath = os.path.join(Config.BASE_DIR, paper_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved metadata to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save metadata JSON: {e}")
            return False
