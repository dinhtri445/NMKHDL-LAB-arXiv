# src/utils.py
import json
import os
import re
import shutil
import nltk
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

# ==========================================
# CẤU HÌNH NLTK (Tự động tải data nếu thiếu)
# ==========================================
def ensure_nltk_data():
    """
    Kiểm tra và tải dữ liệu NLTK an toàn.
    """
    # 1. Tải punkt_tab (cho NLTK bản mới)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except (LookupError, OSError):
        print("Đang tải 'punkt_tab'...")
        nltk.download('punkt_tab', quiet=True)
    
    # 2. Tải punkt (fallback)
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, OSError):
        print("Đang tải 'punkt'...")
        nltk.download('punkt', quiet=True)

# Gọi hàm check ngay khi import file này
ensure_nltk_data()

# ==========================================
# FILE I/O
# ==========================================
def load_json(path):
    """Đọc file JSON an toàn."""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    """Lưu file JSON với format đẹp và encoding utf-8."""
    # Tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def copy_file(src, dst):
    """Copy file từ src sang dst."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def has_tex_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".tex"):
                return True
    return False

def find_files_recursive(root_dir, extension):
    """
    Tìm đệ quy tất cả các file có đuôi extension trong root_dir và thư mục con.
    """
    found_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extension.lower()):
                found_files.append(os.path.join(root, file))
    return found_files

# ==========================================
# TEXT PROCESSING
# ==========================================
def normalize_whitespace(text):
    """Xóa khoảng trắng thừa, đưa về 1 dòng."""
    if not text:
        return ""
    return ' '.join(text.split())

def clean_latex_comments(text):
    """Xóa comment (%) trong LaTeX nhưng không xóa % được escape (\\%)."""
    # Regex này tìm % không đi sau dấu \
    return re.sub(r'(?<!\\)%.*', '', text)

def clean_general_text(text):
    """
    Dùng cho văn bản thông thường
    """
    if not text: return ""

    # ==================================================================
    # GIAI ĐOẠN 0: XỬ LÝ ĐẶC BIỆT CHO CÁC TEMPLATE HỘI NGHỊ 
    # ==================================================================
    
    # 1. Xóa khối \twocolumn[...] chứa title/author (Thường gặp ở ICML/NeurIPS)
    text = re.sub(r'\\twocolumn\s*\[.*?\]', '', text, flags=re.DOTALL)

    # 2. Xóa các lệnh Metadata có tiền tố lạ (icmltitle, aistatsauthor, prlkeywords...)
    meta_keywords = (
        r'title|author|affiliation|affil|email|address|keywords|'
        r'thanks|copyright|notice|correspondence|classification|'
        r'contributions|equal|corresp'
    )
    
    meta_pattern = r'\\[a-zA-Z]*(' + meta_keywords + r')[a-zA-Z]*\{[^}]*\}'
    while re.search(meta_pattern, text, flags=re.IGNORECASE):
        text = re.sub(meta_pattern, '', text, flags=re.IGNORECASE)

    # 3. Xóa các lệnh in ấn đặc biệt (Layout commands)
    text = re.sub(r'\\printAffiliationsAndNotice', '', text)
    text = re.sub(r'\\maketitle', '', text)
    text = re.sub(r'\\(begin|end)\{icmlauthorlist\}', '', text) # Xóa môi trường list author

    # ==================================================================
    # GIAI ĐOẠN 1: XÓA CẤU TRÚC RÁC CƠ BẢN 
    # ==================================================================
    text = re.sub(r'\\begingroup.*?\\endgroup', '', text, flags=re.DOTALL)
    text = re.sub(r'\\footnote(?:text)?(?:\[[^\]]*\])?\{[^}]*\}', '', text)
    text = re.sub(r'\\def\\[a-zA-Z]+(?:#[0-9])?\{[^}]*\}', '', text)

    # ==================================================================
    # GIAI ĐOẠN 2: XÓA THAM CHIẾU & LIÊN KẾT 
    # ==================================================================
    text = re.sub(r'\\cite[a-zA-Z]*(?:\*)?(?:\[[^\]]*\])?\{[^}]+\}', '', text)
    text = re.sub(r'\\(?:ref|eqref|cref|autoref)(?:\[[^\]]*\])?\{[^}]+\}', '', text)
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    text = re.sub(r'\\url\{[^}]+\}', '', text)
    text = re.sub(r'\\href\{[^}]+\}\{[^}]+\}', '', text)
    text = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}', '', text)

    # ==================================================================
    # GIAI ĐOẠN 2.5: XỬ LÝ THAM SỐ ĐẶC THÙ 
    # ==================================================================
    text = re.sub(r'\[\s*(width|height|scale|angle)\s*=[^\]]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\caption\[[^\]]*\]', r'\\caption', text)

    # ==================================================================
    # GIAI ĐOẠN 3: BÓC TÁCH ĐỊNH DẠNG 
    # ==================================================================
    cmds_to_unwrap = (
        r'textbf|textit|texttt|emph|underline|textsc|textsf|textsl|textmd|'
        r'section\*?|subsection\*?|subsubsection\*?|paragraph\*?|subparagraph\*?|'
        r'caption|item|ditem|color|textcolor|scalebox|framebox|mbox'
    )
    pattern = r'\\(?:' + cmds_to_unwrap + r')(?:\[[^\]]*\])?\{([^}]+)\}'
    
    loop_count = 0
    while re.search(pattern, text) and loop_count < 10:
        text = re.sub(pattern, r'\1', text)
        loop_count += 1

    # ==================================================================
    # GIAI ĐOẠN 4: CLEAN KÝ TỰ & LAYOUT 
    # ==================================================================
    text = re.sub(r'\\(?:centering|hfill|vfill|newpage|clearpage|noindent|small|large|huge|normalsize)', '', text)
    text = re.sub(r'\\(?:bigskip|medskip|smallskip|vspace|hspace|vskip)(?:\{[^}]+\}| [0-9\.]+[a-zA-Z]+)?', '', text) 
    text = re.sub(r'\[[htbp!H]+\]', '', text) 

    text = text.replace('~', ' ') 
    text = text.replace(r'\\', ' ')
    text = text.replace(r'\&', '&').replace(r'\%', '%').replace(r'\$', '$')
    text = text.replace(r'\_', '_').replace(r'\#', '#').replace(r'\{', '{').replace(r'\}', '}')

    text = re.sub(r"``|''", '"', text)
    text = text.replace(r"`", "'")
    text = re.sub(r'---?', '-', text)
    text = text.replace('{}', '')

    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'\.{2,}', '.', text)

    return normalize_whitespace(text)

def clean_table_content(text):
    """
    Dùng riêng cho nội dung bên trong Table.
    """
    # Xóa các đường kẻ, vì ta coi table là text node
    text = re.sub(r'\\(top|mid|bottom)rule', '', text)
    text = re.sub(r'\\hline', '', text)
    text = re.sub(r'\\cmidrule[\{\[].*?[\}\]]', '', text)
    
    return normalize_whitespace(text)

def clean_math_content(text):
    """
    Dùng riêng cho Equation.
    """
    # Chỉ xóa các lệnh label (tham chiếu) không hiển thị
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    
    # Xóa \nonumber (format)
    text = re.sub(r'\\nonumber', '', text)
    
    # KHÔNG chạy regex xóa [...] ở đây để bảo vệ x[n]
    return normalize_whitespace(text)

def normalize_math(text):
    r"""
    Chuẩn hóa các định dạng toán học trong LaTeX về 2 dạng chuẩn:
    1. Inline: $ ... $
    2. Block: \begin{equation} ... \end{equation}
    """
    # ---------------------------
    # PHẦN 1: INLINE MATH
    # Target: $ ... $
    # ---------------------------
    
    # Case 1.1: \( ... \) -> $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # Case 1.2: \begin{math} ... \end{math} -> $ ... $ (Ít gặp nhưng nên có)
    text = re.sub(r'\\begin\{math\}(.*?)\\end\{math\}', r'$\1$', text, flags=re.DOTALL)

    # ---------------------------
    # PHẦN 2: BLOCK MATH
    # Target: \begin{equation} ... \end{equation}
    # ---------------------------
    
    def replace_block(match):
        content = match.group(1).strip()
        if not content: return ""
        return f'\\begin{{equation}}{content}\\end{{equation}}'

    # Case 2.1: $$ ... $$ -> Equation
    text = re.sub(r'\$\$(.*?)\$\$', replace_block, text, flags=re.DOTALL)
    
    # Case 2.2: \[ ... \] -> Equation (Đây là Block, KHÔNG phải Inline)
    text = re.sub(r'\\\[(.*?)\\\]', replace_block, text, flags=re.DOTALL)
    
    # Case 2.3: \begin{displaymath} ... \end{displaymath} -> Equation
    text = re.sub(r'\\begin\{displaymath\}(.*?)\\end\{displaymath\}', replace_block, text, flags=re.DOTALL)

    # Case 2.4: \begin{equation*} ... \end{equation*} -> Equation (Bỏ dấu sao)
    # Chỉ xử lý equation*, không xử lý equation thường (để tránh lặp)
    text = re.sub(r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}', replace_block, text, flags=re.DOTALL)

    return text

def split_sentences(text):
    """Dùng NLTK để tách câu chuẩn xác."""
    if not text:
        return []
    
    sentences = nltk.sent_tokenize(text)
    valid_sentences = []
    
    for s in sentences:
        s = normalize_whitespace(s)
        
        # Độ dài phải > 2 ký tự và phải chứa ít nhất 1 chữ cái
        if len(s) > 2 and any(c.isalpha() for c in s):
            valid_sentences.append(s)
            
    return valid_sentences

def generate_paper_element_id(paper_id, elem_type, counter):
    """Tạo ID chuẩn: {PaperID}-{Type}-{Count}"""
    return f"{paper_id}-{elem_type}-{counter}"

# ==========================================
# CÁC HÀM HỖ TRỢ XỬ LÝ BIBTEX 
# ==========================================

def parse_bibtex_entry(bib_string):
    """
    Dùng thư viện bibtexparser để parse nội dung file .bib chuẩn xác.
    """
    parser = BibTexParser()
    
    # 1. Cho phép các type lạ (@online, @software)
    # Mặc định là True (nghĩa là nó sẽ bỏ qua và warning). 
    # Ta set False để nó LẤY LUÔN các entry này.
    parser.ignore_nonstandard_types = False 
    
    # 2. Xử lý Unicode (như cũ)
    parser.customization = convert_to_unicode 
    
    try:
        bib_database = bibtexparser.loads(bib_string, parser=parser)
        return bib_database.entries
    except Exception as e:
        return []

def get_referenced_bib_files(latex_content):
    """
    Tìm danh sách các tên file .bib được gọi trong lệnh \bibliography{...} 
    hoặc \addbibresource{...}.
    Trả về set các tên file (đã bỏ đuôi .bib và chuyển về lowercase).
    """
    referenced = set()
    
    # Pattern 1: \bibliography{file1, file2}
    # Lấy nội dung trong ngoặc
    bib_matches = re.findall(r'\\bibliography\{([^}]+)\}', latex_content, re.IGNORECASE)
    for match in bib_matches:
        # Tách dấu phẩy nếu có nhiều file
        files = match.split(',')
        for f in files:
            name = f.strip().lower()
            if name.endswith('.bib'): name = name[:-4]
            referenced.add(name)
            
    # Pattern 2: \addbibresource{file.bib} (Thường dùng trong BibLaTeX)
    resource_matches = re.findall(r'\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}', latex_content, re.IGNORECASE)
    for match in resource_matches:
        name = match.strip().lower()
        if name.endswith('.bib'): name = name[:-4]
        referenced.add(name)
        
    return referenced

def parse_bibitem_content(key, content):
    """
    Dùng Heuristic (Regex) để tách thông tin từ \bibitem thô.
    Cố gắng lấy: Title, Year, Eprint (arXiv ID).
    """
    # Thay thế các ký tự đặc biệt ngay từ đầu để Regex hoạt động trơn tru
    content = content.replace(r'\textquoteright{}', "'").replace(r'\textquoteright', "'")
    content = content.replace(r'\textendash', "-").replace(r'---', "-").replace(r'--', "-")

    clean_content = re.sub(r'\s+', ' ', content).strip()
    
    # bibtexparser dùng key ENTRYTYPE và ID
    entry = {
        'ENTRYTYPE': 'misc', 
        'ID': key,           
        'note': clean_content
    }

    # 1. Trích xuất arXiv ID (QUAN TRỌNG cho bước 2.2 Matching)
    arxiv_match = re.search(r'(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?', clean_content, re.IGNORECASE)
    if arxiv_match:
        entry['eprint'] = arxiv_match.group(1)
        entry['archivePrefix'] = 'arXiv'

    # 2. Trích xuất Title (Nằm trong \textit, \emph, "", '')
    title_pattern = re.compile(r'(\\(?:textit|emph|it)\s*\{|\{\\it\s+|``|(?<!\\)"|”)(.*?)(?:\}|(?<!\\)\'\'|(?<!\\)"|“)')
    
    # Tìm kiếm trên clean_content để index không bị sai lệch do newline
    title_match = title_pattern.search(clean_content)

    if title_match:
        # Group 2 là nội dung title
        raw_title = title_match.group(2).strip()
        # Clean title: xóa dấu phẩy/chấm cuối câu
        clean_title = re.sub(r'[.,;]+$', '', raw_title).strip()
        entry['title'] = clean_title
        
        # --- LOGIC TÁCH AUTHOR & JOURNAL ---
        start_idx = title_match.start()
        end_idx = title_match.end()

        # A. AUTHOR (Bên trái Title)
        left_part = clean_content[:start_idx].strip()
        # Xóa các lệnh \bibitem{...}
        left_part = re.sub(r'\\bibitem(?:\[[^\]]*\])?\{[^}]+\}', '', left_part).strip()
        # Xóa dấu phẩy/chấm cuối cùng
        left_part = re.sub(r'[.,;:]+$', '', left_part).strip()
        
        if left_part:
            entry['author'] = left_part

        # B. JOURNAL/VOLUME (Bên phải Title)
        right_part = clean_content[end_idx:].strip()
        
        # Xóa dấu phẩy, chấm, khoảng trắng dư thừa Ở ĐẦU chuỗi Journal
        right_part = re.sub(r'^[\s,.:;\}]+', '', right_part)

        # Cờ để chuyển thành @article
        is_article = False
        
        # Tìm Volume (\textbf{123} hoặc {\bf 123})
        vol_match = re.search(r'(?:\\textbf|\\bf)\s*\{?(\d+)\}?', right_part)
        if vol_match:
            entry['volume'] = vol_match.group(1)
            is_article = True

        # Tìm Pages
        page_match = re.search(r',\s*(\d+(?:--\d+)?)', right_part)
        if page_match:
            entry['pages'] = page_match.group(1)

        # Lấy tên Journal: Cắt bỏ phần Volume hoặc Year phía sau
        journal_part = re.split(r'\\textbf|\\bf|\(\d{4}\)', right_part)[0].strip()
        
        # Xóa các lệnh latex bao ngoài (nếu có)
        journal_clean = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', journal_part)
        
        journal_clean = re.sub(r'[.,;]+$', '', journal_clean).strip()
        journal_clean = journal_clean.rstrip('{').strip()
        # Logic lọc rác: Journal phải đủ dài và không phải là arXiv ID
        if len(journal_clean) > 2 and 'arXiv' not in journal_clean:
            entry['journal'] = journal_clean
            is_article = True
            
        if is_article:
            entry['ENTRYTYPE'] = 'article'

    else:
        # Fallback: Không tìm thấy format title
        entry['title_guess'] = clean_content[:100]

    # 3. Trích xuất Year (Tìm số (19xx) - (20xx))
    year_match = re.search(r'\((\d{4})[a-z]?\)', clean_content)
    if year_match:
        y = int(year_match.group(1))
        if 1900 <= y <= 2030:
            entry['year'] = str(y)

    return entry

def extract_specific_bib_entries(bib_content, target_keys):
    """
    Quét thủ công file .bib (dạng text) để tìm các entry cụ thể.
    Nhanh hơn rất nhiều so với parse toàn bộ file lớn.
    
    :param bib_content: Nội dung text của file .bib
    :param target_keys: Set các citation key cần tìm (lowercase)
    :return: List các entry dictionaries (đã parse sơ bộ)
    """
    entries = []
    if not target_keys:
        return entries

    # Regex tìm điểm bắt đầu entry: @article{key, hoặc @book{key,
    # Group 1: Type, Group 2: Key
    entry_start_pattern = re.compile(r'@(\w+)\s*\{\s*([^,]+),', re.IGNORECASE)
    
    lines = bib_content.splitlines()
    current_entry_lines = []
    inside_entry = False
    brace_balance = 0

    for line in lines:
        stripped = line.strip()
        if not stripped: continue

        if not inside_entry:
            # Tìm dòng bắt đầu entry mới
            match = entry_start_pattern.search(line)
            if match:
                raw_key = match.group(2).strip()
                # Kiểm tra xem key này có nằm trong danh sách cần lấy không
                if normalize_title_string(raw_key) in target_keys or raw_key.lower() in target_keys:
                    inside_entry = True
                    current_entry_lines = [line]
                    # Đếm ngoặc nhọn để biết khi nào kết thúc entry
                    brace_balance = line.count('{') - line.count('}')
        else:
            # Đang ở trong entry cần lấy -> Ghi lại
            current_entry_lines.append(line)
            brace_balance += line.count('{') - line.count('}')
            
            # Nếu balance về 0 (hoặc < 0 do lỗi format), coi như kết thúc entry
            if brace_balance <= 0:
                # Gộp dòng lại thành chuỗi bibtex của entry này
                entry_str = "\n".join(current_entry_lines)
                # Parse entry đơn lẻ này bằng thư viện (rất nhanh)
                parsed = parse_bibtex_entry(entry_str)
                if parsed:
                    entries.extend(parsed)
                
                # Reset trạng thái
                inside_entry = False
                current_entry_lines = []

    return entries

def normalize_title_string(s):
    """Chuẩn hóa chuỗi để so sánh trùng lặp (Fingerprint)."""
    if not s: return ""
    # Chuyển về lowercase, bỏ hết ký tự đặc biệt, chỉ giữ chữ và số
    return re.sub(r'[\W_]+', '', s.lower())

def get_active_citations(latex_content):
    """Quét toàn bộ latex để tìm các key được cite."""
    active_keys = set()
    pattern = r'\\cite(?:[a-zA-Z]*\*?)?(?:\[[^\]]*\])?\{([^}]+)\}'
    
    matches = re.findall(pattern, latex_content)
    for m in matches:
        # Split bởi dấu phẩy nếu cite nhiều bài: \cite{key1, key2}
        keys = [k.strip() for k in m.split(',')]
        for k in keys:
            # Lưu cả key gốc và key lowercase để match cho chắc
            active_keys.add(k) 
            active_keys.add(k.lower())
        
    return active_keys


