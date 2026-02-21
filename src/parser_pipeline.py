# src/parser_pipeline.py
import os
import re
from utils import (
    save_json, 
    copy_file, 
    clean_latex_comments, 
    has_tex_files,
    find_files_recursive,
    parse_bibtex_entry, 
    get_referenced_bib_files,
    parse_bibitem_content,
    extract_specific_bib_entries,
    normalize_title_string, 
    get_active_citations
)
from hierarchy_parser import LatexHierarchyParser

import logging

log = logging.getLogger(__name__)

class LatexParser:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.parsed_references = []  # Chứa danh sách các dict reference đã parse
        self.active_citation_keys = set() # Chứa danh sách các key thực sự được cite

    def run_parsing(self):
        """Hàm chạy chính cho từng bài báo."""
        paper_id = os.path.basename(self.input_dir)
        output_paper_path = os.path.join(self.output_dir, paper_id)

        # Dictionary lưu trạng thái để trả về cho main.py thống kê
        stats = {
            "paper_id": paper_id,
            "has_tex": False,
            "hierarchy_success": False,
            "bib_created": False,
            "refs_count": 0,
            "avg_depth": 0,
            "max_depth": 0,
            "avg_branching": 0
        }

        # 1. Copy Metadata & References (Giữ nguyên gốc)
        copy_file(os.path.join(self.input_dir, 'metadata.json'), 
                  os.path.join(output_paper_path, 'metadata.json'))
        copy_file(os.path.join(self.input_dir, 'references.json'), 
                  os.path.join(output_paper_path, 'references.json'))

        tex_dir = os.path.join(self.input_dir, 'tex')
        if not os.path.exists(tex_dir):
            log.warning(f"No tex folder found for {paper_id}")
            return stats

        # Duyệt qua các version (v1, v2...)
        versions = [
            d for d in os.listdir(tex_dir)
            if os.path.isdir(os.path.join(tex_dir, d)) 
            and has_tex_files(os.path.join(tex_dir, d))
        ]
        
        if not versions:
            log.warning(f"No valid tex versions found for {paper_id}")
            return stats
        
        stats["has_tex"] = True
        versions = sorted(versions)
        # Khởi tạo hierarchy parser
        hierarchy_parser = LatexHierarchyParser(paper_id)

        for ver in versions:
            ver_path = os.path.join(tex_dir, ver)
            # --- BƯỚC 2.1.1: GOM FILE ---
            full_latex = self._merge_latex_files(ver_path)

            if not full_latex:
                continue
            # --- BƯỚC 2.1.2: XÂY DỰNG HIERARCHY ---
            # Parse nội dung đã gộp thành cây
            hierarchy_parser.parse_version(ver, full_latex)

            # --- BƯỚC 2.1.3 (Phần Ref): TRÍCH XUẤT BIBTEX ---
            self._extract_bibtex(ver_path, full_latex)

        # Lưu Hierarchy và lấy thống kê cây
        hierarchy_json = hierarchy_parser.build_hierarchy_json()
        if hierarchy_json["hierarchy"]:
            save_json(hierarchy_json, os.path.join(output_paper_path, 'hierarchy.json'))
            stats["hierarchy_success"] = True
            
            # Ghi log thành công
            try:
                tree_stats = hierarchy_parser.get_stats()
                stats.update(tree_stats) # Merge depth, branching vào stats chính
                log.info(f"[{paper_id}] Hierarchy parsed. Depth: {stats['avg_depth']}, MaxDepth: {stats['max_depth']}")
            except Exception as e:
                log.warning(f"[{paper_id}] Warning: Could not calc stats: {e}")
        
        # Lưu BibTeX
        refs_path = os.path.join(output_paper_path, 'refs.bib')
        self._save_bibtex(refs_path)
        
        if os.path.exists(refs_path):
            stats["bib_created"] = True
            stats["refs_count"] = len(self.parsed_references)
            log.info(f"[{paper_id}] refs.bib created with {stats['refs_count']} entries.")

        return stats

    def _merge_latex_files(self, ver_path):
        """Tìm file main và gộp các file con."""
        # 1. Tìm file main (có chứa \documentclass)
        main_file = None
        for root, dirs, files in os.walk(ver_path):
            for file in files:
                if file.endswith('.tex'):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            if r'\documentclass' in content and r'\begin{document}' in content and r'\end{document}' in content:
                                main_file = full_path
                                break
                    except Exception:
                        continue
            if main_file: break
        
        if not main_file:
            log.warning(f"No main .tex file found in {ver_path}")
            return ""

        # 2. Đệ quy gộp file
        return self._recursive_read(main_file, ver_path)

    def _recursive_read(self, file_path, root_ver_path, visited=None):
        """Đệ quy đọc và thay thế \\input, \\include."""
        if not os.path.exists(file_path):
            log.warning(f"File not found: {file_path}")
            return ""
        
        # Xử lý trường hợp file A input B và ngược lại
        if visited is None:
            visited = set()

        file_path = os.path.abspath(file_path)
        if file_path in visited:
            log.warning(f"Circular input detected: {file_path}")
            return ""

        visited.add(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Xóa comment trước khi xử lý
            content = clean_latex_comments(content)

            # Regex tìm \input{...} hoặc \include{...}
            # Group 1: tên file
            pattern = r'\\(?:input|include)\{([^}]+)\}'
            
            def replace_match(match):
                sub_filename = match.group(1)
                if not sub_filename.endswith('.tex'):
                    sub_filename += '.tex'
                
                # Xử lý đường dẫn tương đối
                dir_current = os.path.dirname(file_path)
                sub_path = os.path.join(dir_current, sub_filename)
                
                # Nếu không thấy, thử tìm từ root version folder
                if not os.path.exists(sub_path):
                    sub_path = os.path.join(root_ver_path, sub_filename)
                
                return self._recursive_read(sub_path, root_ver_path, visited)
            
            content = re.sub(pattern, replace_match, content)
        finally:
            # Backtrack: remove dù có lỗi hay không
            if file_path in visited:
                visited.remove(file_path)

        return content
    
    def _extract_bibtex(self, ver_path, full_latex):
        """
        Trích xuất và Parse References thành Dictionary.
        """
        # Set theo dõi các key đã được parse trong version này để tránh trùng lặp nguồn
        # (Ưu tiên lấy từ .bib, nếu có rồi thì \bibitem không lấy nữa)
        local_seen_keys = set()

        # 1. Xác định các Active Keys (những bài thực sự được cite)
        # Lưu vào biến instance để dùng ở bước save
        
        current_active = get_active_citations(full_latex)
        self.active_citation_keys.update(current_active)
        
        # Chuẩn hóa active keys để so sánh (lowercase)
        target_keys_normalized = {k.lower() for k in self.active_citation_keys}

        # Lấy danh sách các file bib được gọi
        used_bib_names = get_referenced_bib_files(full_latex)
        # 2. Ưu tiên 1: Parse từ file .bib (nếu có)
        bib_files = find_files_recursive(ver_path, '.bib')
        if bib_files:
            for bib_file in bib_files:
                # Lấy tên file để kiểm tra (vd: /path/to/custom.bib -> custom)
                fname = os.path.basename(bib_file)
                fname_no_ext = os.path.splitext(fname)[0].lower()
                
                # Nếu file này không nằm trong danh sách được gọi -> BỎ QUA
                # (Tránh parse file 50MB bị comment out)
                if used_bib_names and (fname_no_ext not in used_bib_names):
                    continue

                try:
                    file_size = os.path.getsize(bib_file)
                    with open(bib_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    entries = []
                    if file_size > 2 * 1024 * 1024: 
                        log.info(f"Large Bib file detected ({file_size/1024/1024:.2f} MB): {fname}. Using fast extraction.")
                        entries = extract_specific_bib_entries(content, target_keys_normalized)
                    else:
                        entries = parse_bibtex_entry(content)
    
                    for entry in entries:
                        # bibtexparser trả về dict, key ID là citation key
                        local_seen_keys.add(entry['ID'])
                        self.parsed_references.append(entry)
                    
                except Exception as e:
                    log.warning(f"Error parsing bib file {bib_file}: {e}")

        # 3. Ưu tiên 2: Convert từ \bibitem (nếu không có .bib hoặc bổ sung)
        pattern = r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{|\\bibliography|\Z)'
        matches = re.findall(pattern, full_latex, re.DOTALL)
        
        for key, content in matches:
            # Nếu key này đã lấy được từ file .bib chuẩn rồi thì bỏ qua \bibitem
            if key in local_seen_keys: continue
            entry = parse_bibitem_content(key, content)
            self.parsed_references.append(entry)

    def _save_bibtex(self, path):
        """
        Xử lý Deduplication, Unionization, Filter và Lưu file .bib.
        """
        if not hasattr(self, 'parsed_references') or not self.parsed_references:
            return

        # 1. DEDUPLICATION & UNIONIZATION LOGIC
        unique_map = {} # Map: Fingerprint -> Entry Data
        key_mapping = {} # Map: Old Key -> New Canonical Key

        for entry in self.parsed_references:
            # Tạo Fingerprint
            title = entry.get('title') or entry.get('note') or entry.get('title_guess') or ""
            
            fingerprint = normalize_title_string(title)
            
            if len(fingerprint) < 10:
                fingerprint = f"RAW_{entry['ID']}"

            if fingerprint in unique_map:
                # Unionize: Merge thông tin
                existing = unique_map[fingerprint]
                for k, v in entry.items():
                    # Logic merge: Lấy value dài hơn
                    if k not in existing or (len(str(v)) > len(str(existing[k]))):
                        existing[k] = v
                key_mapping[entry['ID']] = existing['ID']
            else:
                unique_map[fingerprint] = entry
                key_mapping[entry['ID']] = entry['ID']

        # 2. FILTERING LOGIC [TỐI ƯU HÓA]
        # Thay vì loop lồng nhau, ta xác định ngay những Canonical Key nào cần giữ lại
        keys_to_keep = set()

        for raw_cited_key in self.active_citation_keys:
            # Nếu key được cite có nằm trong map (tức là ta đã parse được nó)
            if raw_cited_key in key_mapping:
                # Lấy key chuẩn (canonical) của nó
                canonical = key_mapping[raw_cited_key]
                keys_to_keep.add(canonical)
        
        # Chỉ giữ lại các entry có key nằm trong keys_to_keep
        final_entries = [entry for entry in unique_map.values() if entry['ID'] in keys_to_keep]

        if not final_entries:
            log.info(f"Skipping {path}: No cited references found.")
            return
        
        # 3. WRITE TO FILE
        with open(path, 'w', encoding='utf-8') as f:
            for entry in final_entries:
                etype = entry.get('ENTRYTYPE', 'misc').lower()
                ekey = entry.get('ID', 'unknown')
                
                f.write(f"@{etype}{{{ekey},\n")
                
                for k, v in entry.items():
                    # Bỏ qua các field nội bộ
                    if k in ['ENTRYTYPE', 'ID', 'title_guess']: continue
                    
                    # Escape dấu ngoặc kép
                    v_str = str(v).replace('\\', '\\\\').replace('"', '\\"')
                    f.write(f"  {k.lower()} = {{{v_str}}},\n")
                
                f.write("}\n\n")