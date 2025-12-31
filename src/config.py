import os
import psutil
import json

class Config:
    
    STUDENT_ID = "23120377"
    BASE_DIR = f"./{STUDENT_ID}"
    STATE_FILE = os.path.join(BASE_DIR, "scraper_state.json")
    
    # Khoảng ID bài báo
    START_ID = 4361
    END_ID = 9360
    YYMM = "2406"
    
    # Rate limiting 
    DELAY_BETWEEN_REQUESTS = 1.2  # seconds
    SEMANTIC_SCHOLAR_DELAY = 4  # seconds
    
    # Version handling
    MAX_VERSIONS = 50
    
    # API endpoints
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}"
  
    try:
        _PROCESS = psutil.Process()
        _START_PROCESS_RSS = _PROCESS.memory_info().rss
    except Exception:
        _PROCESS = None
        _START_PROCESS_RSS = 0

    @classmethod
    def update_student_id(cls, student_id: str):
        cls.STUDENT_ID = student_id
        cls.BASE_DIR = f"./{student_id}"
        cls.STATE_FILE = os.path.join(cls.BASE_DIR, "scraper_state.json")

    @classmethod
    def get_memory_usage(cls):
        try:
            if cls._PROCESS:
                return cls._PROCESS.memory_info().rss - cls._START_PROCESS_RSS
            else:
                return psutil.Process(os.getpid()).memory_info().rss - cls._START_PROCESS_RSS
        except Exception:
            return 0
    
    @classmethod
    def save_state(cls, state):
        try:
            os.makedirs(os.path.dirname(cls.STATE_FILE), exist_ok=True)
            with open(cls.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving state: {e}")
    
    @classmethod
    def load_state(cls):
        try:
            if os.path.exists(cls.STATE_FILE):
                with open(cls.STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ Error loading state: {e}")
        return None