# src/cleaning.py
import re
import unicodedata

class ReferenceCleaner:
    """
    Làm sạch và chuẩn hóa dữ liệu trích dẫn 
    (Title, Authors, Date) phục vụ cho Matching Pipeline.
    """

    def __init__(self):
        # Danh sách stop-words tiếng Anh phổ biến trong tên bài báo
        self.stop_words = {
            "a", "an", "the", "on", "in", "at", "of", "for", "to", "and", "or", 
            "with", "by", "from", "as", "is", "are", "was", "were", "be", 
            "that", "which", "how", "what", "about", "using", "via", "based"
        }

    def _remove_latex_commands(self, text):
        """
        Loại bỏ các lệnh LaTeX còn sót lại. 
        VD: \textit{Title} -> Title, \textbf{A} -> A
        """
        if not text: return ""
        # Xóa các command dạng \cmd{content} -> content
        # Loop để xử lý lồng nhau: \textbf{\textit{Title}}
        while r'\\' in text or r'{' in text:
            
            prev_text = text
            
            text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)
            
            text = re.sub(r'\{([^}]+)\}', r'\1', text)
            
            text = re.sub(r'\\[a-zA-Z]+', ' ', text)
            
            # Nếu sau 1 vòng không đổi gì thì break để tránh lặp vô tận
            if text == prev_text: 
                text = text.replace('\\', '').replace('{', '').replace('}', '')
                break
        return text

    def _normalize_unicode(self, text):
        """
        Chuyển các ký tự có dấu về dạng ASCII.
        VD: "Schrödinger" -> "schrodinger", "Café" -> "cafe"
        """
        if not text: return ""
        text = unicodedata.normalize('NFD', text)
        text = "".join([c for c in text if unicodedata.category(c) != 'Mn'])
        return text

    def clean_text_basic(self, text):
        """
        Làm sạch cơ bản: Lowercase + bỏ dấu câu + bỏ khoảng trắng thừa.
        """
        if not text: return ""
        
        # 1. Bỏ LaTeX commands trước
        text = self._remove_latex_commands(text)
        
        # 2. Unicode normalization
        text = self._normalize_unicode(text)
        
        # 3. Lowercase
        text = text.lower()
        
        # 4. Thay thế các ký tự không phải chữ/số thành khoảng trắng
        # Giữ lại a-z, 0-9. Bỏ hết chấm, phẩy, gạch ngang...
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 5. Xóa khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text

    def clean_title(self, title):
        """
        Làm sạch Title: Basic Clean + Stopword Removal.
        """
        text = self.clean_text_basic(title)
        
        tokens = text.split()
        
        # Lọc stop words (chỉ giữ lại từ có ý nghĩa)
        filtered_tokens = [t for t in tokens if t not in self.stop_words]
        
        if not filtered_tokens:
            return text
            
        return " ".join(filtered_tokens)

    def clean_authors(self, authors_input):
        """
        Chuẩn hóa danh sách tác giả về dạng set các tên đã đơn giản hóa.
        """
        author_set = set()
        
        # 1. Chuẩn hóa Input về dạng List
        raw_list = []
        if isinstance(authors_input, list):
            raw_list = authors_input
        elif isinstance(authors_input, str):
            # Tách bởi " and " (chuẩn BibTeX)
            raw_list = authors_input.split(" and ")
        else:
            return set()

        for name in raw_list:
            # Clean cơ bản
            clean_name = self.clean_text_basic(name)
            
            tokens = sorted(clean_name.split())
            normalized_name = " ".join(tokens)
            
            # Bỏ các tên quá ngắn
            if len(normalized_name) > 1:
                author_set.add(normalized_name)
                
        return author_set

    def extract_year(self, date_input):
        """
        Lấy năm từ chuỗi ngày tháng bất kỳ.
        """
        if not date_input: return None
        
        # Regex tìm 4 chữ số liên tiếp (19xx hoặc 20xx)
        match = re.search(r'(19|20)\d{2}', str(date_input))
        if match:
            return int(match.group(0))
        return None
