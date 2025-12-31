
import arxiv
import time
import os
import glob
import tarfile
import logging
import threading
import tempfile
import shutil
import traceback
from typing import List, Dict, Any
from config import Config
import certifi
import socket

os.environ['SSL_CERT_FILE'] = certifi.where()

logger = logging.getLogger('SourceDownloader')

class RateLimiter:
    """
    Bộ giới hạn tốc độ toàn cục (an toàn cho luồng) áp dụng khoảng thời gian tối thiểu
    giữa các yêu cầu liên tiếp trên tất cả các luồng.
    """
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed:
                to_sleep = self.next_allowed - now
                logger.debug(f"RateLimiter sleeping {to_sleep:.3f}s")
                time.sleep(to_sleep)
            # update next 
            self.next_allowed = time.time() + self.min_interval

# Global rate limiter
GLOBAL_RATE_LIMITER = RateLimiter(min_interval=Config.DELAY_BETWEEN_REQUESTS or 1.0)

class SourceDownloader:
    def __init__(self, client=None, max_retries: int = 3, extract_retries: int = 3):
        self.client = client or arxiv.Client()
        self.logger = logging.getLogger('SourceDownloader')
        self.max_retries = max_retries
        self.extract_retries = extract_retries

    def _safe_request(self, func, *args, **kwargs):
        """
        Thực hiện yêu cầu arXiv với giới hạn tốc độ toàn cục và thử lại.
        """
        attempt = 0
        while attempt <= self.max_retries:
            try:
                GLOBAL_RATE_LIMITER.wait()
                return func(*args, **kwargs)
            except (arxiv.HTTPError, socket.timeout, ConnectionError) as e:
                attempt += 1
                wait = 2 ** attempt
                self.logger.warning(f"Transient error on arXiv request (attempt {attempt}) -> {e}. Backing off {wait}s")
                time.sleep(wait)
            except Exception as e:
                self.logger.error(f"Non-recoverable error when requesting arXiv: {e}")
                self.logger.debug(traceback.format_exc())
                raise
        raise RuntimeError("Max retries exceeded for arXiv request")

    def download_all_versions(self, base_id: str) -> List[Dict[str, Any]]:
        """
        Tải xuống tất cả các phiên bản khả dụng cho một base_id
        """
        versions = []
        total_original = 0
        total_extracted = 0
        paper_folder = base_id.replace('.', '-')
        for v in range(1, Config.MAX_VERSIONS + 1):
            arxiv_id = f"{base_id}v{v}"
            self.logger.info(f"Attempting to download {arxiv_id}")
            try:
                # get paper metadata
                def get_paper():
                    search = arxiv.Search(id_list=[arxiv_id], max_results=1)
                    return next(self.client.results(search), None)
                paper = self._safe_request(get_paper)

                if paper is None:
                    self.logger.info(f"No version {v} for {base_id}; stopping.")
                    break

                version_folder_name = f"{paper_folder}v{v}"
                version_dir = os.path.join(Config.BASE_DIR, paper_folder, "tex", version_folder_name)
                os.makedirs(version_dir, exist_ok=True)

                # download + validate + extract with retries
                success, orig_size, ex_size = self._download_validate_and_extract(paper, arxiv_id, version_dir)
                if success:
                    versions.append({
                        "version": v,
                        "arxiv_id": arxiv_id,
                        "title": getattr(paper, "title", None),
                        "published": getattr(paper, "published", None).isoformat() if getattr(paper, "published", None) else None,
                        "updated": getattr(paper, "updated", None).isoformat() if getattr(paper, "updated", None) else None,
                    })
                    total_original += orig_size or 0
                    total_extracted += ex_size or 0
                else:
                    self.logger.warning(f"Failed to obtain valid source for {arxiv_id}; continuing to next version.")

            except Exception as e:
                self.logger.warning(f"Error handling {arxiv_id}: {e}")
                self.logger.debug(traceback.format_exc())
                continue

        
        try:
            self.recover_unextracted_tars(paper_folder)
        except Exception:
            self.logger.debug("recover_unextracted_tars encountered an error", exc_info=True)

        return versions, total_original, total_extracted

    def _download_validate_and_extract(self, paper, arxiv_id: str, version_dir: str) -> bool:
        """
        Tải xuống kho lưu trữ nguồn, xác thực gzip/tar, sau đó giải nén .tex/.bib vào version_dir.
        """
        filename = f"{arxiv_id}.tar.gz"
        saved_path = None
        attempt = 0
        while attempt <= self.max_retries:
            try:
                GLOBAL_RATE_LIMITER.wait()
                saved_path = paper.download_source(dirpath=version_dir, filename=filename)
                if not saved_path or not os.path.exists(saved_path):
                    raise RuntimeError("download_source returned no file")
                size = os.path.getsize(saved_path)
                if size < 1024:  # threshold: các tập tin rất nhỏ có khả năng không hợp lệ
                    raise RuntimeError(f"Downloaded file too small ({size} bytes)")
                
                if not self._is_gzip_file(saved_path):
                    raise RuntimeError("Downloaded file is not a valid gzip file")
                # tar test
                if not tarfile.is_tarfile(saved_path):
                    raise RuntimeError("Downloaded file is not a valid tar archive")
                original_size = os.path.getsize(saved_path)
                
                # extract with retries
                extracted = self._extract_tex_bib_with_retries(saved_path, version_dir, retries=self.extract_retries)
                
                extracted_size = 0
                if extracted:
                    for root, dirs, files in os.walk(version_dir):
                        for f in files:
                            try:
                                extracted_size += os.path.getsize(os.path.join(root, f))
                            except Exception:
                                pass
                        
                if not extracted:
                    raise RuntimeError("Extraction failed after retries")
                # success
                return True, original_size, extracted_size

            except Exception as e:
                attempt += 1
                wait = 2 ** attempt
                self.logger.warning(f"download/validate/extract failed for {arxiv_id} (attempt {attempt}): {e}. Retrying in {wait}s")
                self.logger.debug(traceback.format_exc())
                # remove suspect file before retrying
                try:
                    if saved_path and os.path.exists(saved_path):
                        os.remove(saved_path)
                except Exception:
                    pass
                time.sleep(wait)
                continue

        self.logger.error(f"Giving up download for {arxiv_id} after {self.max_retries} attempts")
        return False, 0, 0

    def _is_gzip_file(self, path: str) -> bool:
        """
        Kiểm tra nhanh xem tệp có tiêu đề gzip không.
        """
        try:
            with open(path, 'rb') as f:
                magic = f.read(2)
            return magic == b'\x1f\x8b'
        except Exception:
            return False

    def _extract_tex_bib_with_retries(self, tar_path: str, version_dir: str, retries: int = 3) -> bool:
        """
        Chỉ trích xuất .tex và .bib (và các thư mục) từ tar_path vào version_dir.
        """
        attempt = 0
        while attempt < retries:
            tmp_extract_dir = None
            try:
                if not tarfile.is_tarfile(tar_path):
                    raise tarfile.ReadError("Not a tar archive")

                tmp_extract_dir = tempfile.mkdtemp(prefix="extract_tmp_", dir=version_dir)
                with tarfile.open(tar_path, 'r:gz') as tar:
                    members = []
                    for m in tar.getmembers():
                        name_lower = m.name.lower()
                        if m.isdir() or name_lower.endswith((".tex", ".bib")):
                            members.append(m)

                    if not members:                       
                        try:
                            os.remove(tar_path)
                        except Exception:
                            self.logger.debug(f"Could not remove tar file {tar_path} after empty extract")
                        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                        return True

                    tar.extractall(path=tmp_extract_dir, members=members)

                # di chuyển nội dung tmp vào cấu trúc bảo toàn version_dir
                for root, dirs, files in os.walk(tmp_extract_dir):
                    rel = os.path.relpath(root, tmp_extract_dir)
                    target_root = os.path.join(version_dir, rel) if rel != "." else version_dir
                    os.makedirs(target_root, exist_ok=True)
                    for f in files:
                        src = os.path.join(root, f)
                        dst = os.path.join(target_root, f)
                        try:
                            os.replace(src, dst)
                        except Exception:
                            shutil.move(src, dst)

                shutil.rmtree(tmp_extract_dir, ignore_errors=True)

                # xóa file tar sau khi giải nén thành công
                try:
                    os.remove(tar_path)
                except Exception:
                    self.logger.debug(f"Could not remove tar file {tar_path} after extraction")

                return True

            except (tarfile.ReadError, EOFError, OSError) as e:
                attempt += 1
                wait = 2 ** attempt
                self.logger.warning(f"Extraction attempt {attempt} failed for {tar_path}: {e}. Retrying in {wait}s")
                self.logger.debug(traceback.format_exc())
                # dọn dẹp tạm thời và xóa tar đáng ngờ để có thể tải lại
                try:
                    if tmp_extract_dir and os.path.exists(tmp_extract_dir):
                        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                except Exception:
                    pass
                time.sleep(wait)
                continue
            except Exception as e:
                self.logger.error(f"Non-recoverable extraction error for {tar_path}: {e}")
                self.logger.debug(traceback.format_exc())
                try:
                    if tmp_extract_dir and os.path.exists(tmp_extract_dir):
                        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                except Exception:
                    pass
                return False

        self.logger.error(f"Extraction failed for {tar_path} after {retries} attempts")
        return False

    def recover_unextracted_tars(self, paper_folder: str):
        """
        Scan BASE_DIR/<paper_folder>/tex/** để tìm bất kỳ *.tar.gz nào còn sót lại và thử giải nén chúng.
        """
        base_path = os.path.join(Config.BASE_DIR, paper_folder, "tex")
        pattern = os.path.join(base_path, "**", "*.tar.gz")
        tar_files = glob.glob(pattern, recursive=True)
        if not tar_files:
            return

        self.logger.info(f"Found {len(tar_files)} leftover tar.gz files under {base_path}, attempting recovery.")
        for tar_path in tar_files:
            try:
                
                version_dir = os.path.dirname(tar_path)
                
                if not os.path.exists(tar_path):
                    continue
                
                if os.path.getsize(tar_path) < 1024 or not self._is_gzip_file(tar_path) or not tarfile.is_tarfile(tar_path):
                    self.logger.warning(f"Leftover {tar_path} seems invalid; removing and skipping.")
                    try:
                        os.remove(tar_path)
                    except Exception:
                        pass
                    continue
                extracted = self._extract_tex_bib_with_retries(tar_path, version_dir, retries=self.extract_retries)
                if extracted:
                    self.logger.info(f"Recovered and extracted {tar_path}")
                else:
                    self.logger.warning(f"Could not extract leftover {tar_path}; removed.")
            except Exception as e:
                self.logger.warning(f"Error recovering {tar_path}: {e}")
                self.logger.debug(traceback.format_exc())
                try:
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                except Exception:
                    pass
