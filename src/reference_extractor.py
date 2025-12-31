import requests
import time
import logging
from typing import Dict, Any
from config import Config
import os
import json
import threading

class RateLimiter:
    """
    Bộ giới hạn tốc độ toàn cục an toàn luồng: đảm bảo khoảng thời gian tối thiểu
    giữa các yêu cầu trên tất cả các worker.
    """
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed:
                delay = self.next_allowed - now
                time.sleep(delay)
            self.next_allowed = time.time() + self.min_interval


# GLOBAL S2 limiter (shared by all workers)
GLOBAL_S2_RATE_LIMITER = RateLimiter(min_interval=max(Config.SEMANTIC_SCHOLAR_DELAY, 1.0))


class ReferenceExtractor:

    def __init__(self):
        self.logger = logging.getLogger("Reference Extractor")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "arXiv-lab-bot/1.0"
        })

    def extract_references(self, arxiv_id_base: str) -> Dict[str, Dict[str, Any]]:
        base_id = arxiv_id_base.split("v")[0] if "v" in arxiv_id_base else arxiv_id_base
        self.logger.info(f"Extracting references for {base_id}")

        url = Config.SEMANTIC_SCHOLAR_URL.format(base_id)
        params = {
            "fields": "references.externalIds,references.title,"
                      "references.authors,references.publicationDate,"
                      "references.paperId"
        }

        try:
            # apply global rate limit
            GLOBAL_S2_RATE_LIMITER.wait()

            r = self.session.get(url, params=params, timeout=30)

            if r.status_code == 429:
                self.logger.warning("Rate limited by Semantic Scholar. Backoff 30s")
                time.sleep(30)
                return self.extract_references(base_id)

            if r.status_code != 200:
                self.logger.warning(
                    f"Semantic Scholar returned code {r.status_code} for {base_id}"
                )
                return {}

            data = r.json()
            references = data.get("references", [])
            return self._process_references(references)

        except requests.RequestException as e:
            self.logger.error(f"Request error for {base_id}: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error for {base_id}: {e}")
            return {}

    def _process_references(self, references) -> Dict[str, Dict[str, Any]]:
        out = {}

        for ref in references:
            ext_ids = ref.get("externalIds", {}) or {}
            arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("arXiv")
            if not arxiv_id:
                continue

            base_arxiv = arxiv_id.split("v")[0]
            key = base_arxiv.replace(".", "-")

            authors_list = [a.get("name") for a in (ref.get("authors") or [])]

            out[key] = {
                "paper_title": ref.get("title"),
                "authors": authors_list,
                "submission_date": ref.get("publicationDate"),
                "semantic_scholar_id": ref.get("paperId")
            }

        return out

    def save_references_json(self, references: Dict[str, Any], paper_folder: str = "", filename: str = "references.json") -> bool:
        try:
            os.makedirs(os.path.join(Config.BASE_DIR, paper_folder), exist_ok=True)
            filepath = os.path.join(Config.BASE_DIR, paper_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(references, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved references to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save references JSON: {e}")
            return False
