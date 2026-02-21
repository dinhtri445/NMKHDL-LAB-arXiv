# src/hierarchy_parser.py
import re
from collections import defaultdict
from utils import ( 
    normalize_math, 
    split_sentences,
    generate_paper_element_id,
    clean_general_text, 
    clean_table_content, 
    clean_math_content
)

class LatexHierarchyParser:
    # Định nghĩa cấp độ sâu của cây
    LEVEL_MAP = {
        "document": 0,
        "abstract": 1,
        "chapter": 1,
        "section": 1,
        "subsection": 2,
        "subsubsection": 3,
        "paragraph": 4,
        "itemize": 5,       # Môi trường itemize là cha
        "item": 6,          # Từng item là con
        "sentence": 7,      # Lá
        "equation": 7,      # Lá (Block formula)
        "figure": 7,        # Lá (Bao gồm cả Table)
        "table": 7          # Table cũng tính là Figure (theo yêu cầu)
    }

    def __init__(self, paper_id):
        self.paper_id = paper_id
        
        # 1. Kho chứa elements toàn cục (Dedup)
        # Key: ID, Value: Content text
        self.elements = {} 
        
        # 2. Map nội dung -> ID (Dùng để kiểm tra trùng lặp)
        # Key: Cleaned Content, Value: ID
        self.content_to_id = {}
        
        # 3. Bộ đếm riêng cho từng loại element trong paper này
        self.counters = defaultdict(int)
        
        # 4. Cấu trúc cây phân cấp theo version
        # Format: {"1": {child_id: parent_id}, "2": ...}
        self.hierarchy = {}

        # Khởi tạo Root Node cho paper
        self.root_id = f"{paper_id}-root"
        self.elements[self.root_id] = "Document Root"
        # Root không cần content_to_id vì nó là unique

    def _get_or_create_id(self, elem_type, content):
        """
        Logic Deduplication:
        - Nếu nội dung đã có trong kho -> Trả về ID cũ.
        - Nếu chưa -> Tạo ID mới, lưu vào kho.
        """
        
        CONTAINER_TYPES = [
            "document", "abstract", "chapter", "section", "subsection", 
            "subsubsection", "paragraph", "itemize_begin", "itemize_end", "item"
        ]

        # 1. Nếu là Container Node -> Luôn tạo mới (Không bao giờ Dedup)
        if elem_type in CONTAINER_TYPES:
            self.counters[elem_type] += 1
            new_id = generate_paper_element_id(self.paper_id, elem_type, self.counters[elem_type])
            self.elements[new_id] = content if content else ""
            # Không lưu vào content_to_id
            return new_id

        # 2. Nếu là Leaf Node (Sentence, Equation, Figure) -> Deduplicate
        if not content:
            # Nội dung rỗng thì tạo mới cho an toàn
            self.counters[elem_type] += 1
            new_id = generate_paper_element_id(self.paper_id, elem_type, self.counters[elem_type])
            self.elements[new_id] = ""
            return new_id

        # Kiểm tra kho Dedup
        if content in self.content_to_id:
            return self.content_to_id[content]

        # Tạo mới Leaf Node và lưu vào kho Dedup
        self.counters[elem_type] += 1
        new_id = generate_paper_element_id(self.paper_id, elem_type, self.counters[elem_type])
        
        self.elements[new_id] = content
        self.content_to_id[content] = new_id
        return new_id

    def _tokenize(self, latex_content):
        """
        Chiến lược: Quét tìm các "Block" đặc biệt trước, sau đó xử lý text còn dư.
        """
        tokens = []
        
        # 1. Danh sách các Pattern cần bắt (Thứ tự quan trọng)
        # Các block lớn (Table, Figure, Equation) cần bắt trọn vẹn để không bị cắt lẻ.
        patterns = [
            # Exclusions: References (Cắt bỏ trước hoặc bỏ qua) -> Xử lý ở hàm parse_version
            (r'\\begin\{abstract\}(.*?)\\end\{abstract\}', "abstract"),

            # Structural Headers
            (r'\\chapter\*?\{([^}]+)\}', "chapter"),
            (r'\\section\*?\{([^}]+)\}', "section"),
            (r'\\subsection\*?\{([^}]+)\}', "subsection"),
            (r'\\subsubsection\*?\{([^}]+)\}', "subsubsection"),
            (r'\\paragraph\*?\{([^}]+)\}', "paragraph"),
            
            # Environments (Containers & Leafs)
            # Itemize/Enumerate: Bắt điểm bắt đầu và kết thúc
            (r'\\begin\{itemize\}', "itemize_begin"), 
            (r'\\end\{itemize\}', "itemize_end"),
            (r'\\begin\{enumerate\}', "itemize_begin"), # Coi enumerate như itemize
            (r'\\end\{enumerate\}', "itemize_end"),
            
            # Items
            (r'\\item\s', "item"), # \item thường không có {} bao nội dung ngay
            
            # Block Leafs (Equation, Figure, Table)
            
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', "equation"),
            
            (r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', "figure"),
            (r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', "figure"),
        ]

        # Tìm tất cả matches và vị trí của chúng
        matches = []
        for pat, type_ in patterns:
            for m in re.finditer(pat, latex_content, re.DOTALL):
                matches.append({
                    "type": type_,
                    "content": m.group(1) if type_ not in ["itemize_begin", "itemize_end", "item"] else "",
                    "start": m.start(),
                    "end": m.end(),
                    "full_match": m.group(0)
                })
        
        # Sắp xếp các match theo vị trí xuất hiện
        matches.sort(key=lambda x: x["start"])

        
        last_end = 0
        
        for m in matches:
            if m["start"] < last_end:
                continue 
            
            # 1. XỬ LÝ TEXT GAP (Khoảng giữa các block)
            # Phần text nằm giữa match trước và match này -> Là Sentences
            text_chunk = latex_content[last_end:m["start"]]
            # Chỉ clean text thường ở đây
            text_chunk = clean_general_text(text_chunk)
            if text_chunk:
                sentences = split_sentences(text_chunk)
                for s in sentences:
                    tokens.append(("sentence", s))

            # 2. XỬ LÝ BLOCK CONTENT (Dùng clean theo ngữ cảnh)
            raw_content = m["content"]
            token_type = m["type"]
            final_content = raw_content # Mặc định

            # --- LOGIC CLEANING---
            if token_type == "abstract":
                # Bước 1: Tạo node cha là Abstract
                tokens.append(("abstract", "Abstract"))
                
                # Bước 2: Xử lý nội dung bên trong thành các câu con
                # Abstract chứa văn bản thường -> dùng clean_general_text
                inner_text = clean_general_text(raw_content)
                if inner_text:
                    sentences = split_sentences(inner_text)
                    for s in sentences:
                        # Các câu này sẽ tự động được Stack xếp vào làm con của "abstract"
                        tokens.append(("sentence", s))
                
                # Cập nhật last_end và continue để không chạy xuống logic mặc định bên dưới
                last_end = m["end"]
                continue

            elif token_type == "equation":
                cleaned_math = clean_math_content(raw_content)
                final_content = f"\\begin{{equation}}{cleaned_math}\\end{{equation}}"
            
            elif token_type == "figure":
               
                if "tabular" in m["full_match"] or "\\begin{table" in m["full_match"]:  
                    final_content = clean_table_content(raw_content)   
                else:
                    final_content = clean_general_text(raw_content)

            elif token_type in ["section", "chapter", "subsection", "subsubsection", "paragraph"]:
                # Tiêu đề -> Coi như text thường
                final_content = clean_general_text(raw_content)
                
            elif token_type == "item":
                final_content = "Item" 
            elif token_type == "itemize_begin":
                final_content = "List"
            elif token_type == "itemize_end":
                final_content = ""
            
            # Append token đã xử lý sạch sẽ
            tokens.append((token_type, final_content))
            
            last_end = m["end"]

        # 3. XỬ LÝ PHẦN DƯ CUỐI CÙNG -> Dùng clean_general_text
        remaining_text = latex_content[last_end:]
        remaining_text = clean_general_text(remaining_text)
        
        if remaining_text:
            sentences = split_sentences(remaining_text)
            for s in sentences:
                tokens.append(("sentence", s))

        return tokens

    def parse_version(self, version_id, latex_content):
        # 1. Normalize Math toàn bộ văn bản trước
        latex_content = normalize_math(latex_content)

        # 2. Xóa References
        # Mục đích: Giữ lại Appendix hoặc nội dung phía sau nếu có
        
        latex_content = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', latex_content, flags=re.DOTALL)
        
        
        latex_content = re.sub(r'\\bibliography\{[^}]+\}', '', latex_content)
      
        latex_content = re.sub(r'\\printbibliography(?:\[[^\]]*\])?', '', latex_content)

        latex_content = re.sub(r'\\section\*?\{(?:References|Bibliography|References\.)\}', '', latex_content, flags=re.IGNORECASE)
        
        # 3. Chỉ lấy nội dung trong \begin{document} (nếu có)
        doc_pattern = r'\\begin\s*\{document\}(.*?)\\end\s*\{document\}'
        doc_match = re.search(doc_pattern, latex_content, re.DOTALL)
        
        if doc_match:
            # Trường hợp đẹp: Tìm thấy cả mở và đóng
            latex_content = doc_match.group(1)
        else:
            # Trường hợp dự phòng: 
            # Ta sẽ cưỡng ép cắt từ \begin{document} trở đi.
            if r'\begin{document}' in latex_content:
                parts = latex_content.split(r'\begin{document}', 1)
                if len(parts) > 1:
                    latex_content = parts[1] # Lấy phần sau begin
                    # Cố gắng cắt phần end nếu có
                    if r'\end{document}' in latex_content:
                        latex_content = latex_content.split(r'\end{document}', 1)[0]

        # 4. Tokenize
        tokens = self._tokenize(latex_content)

        # 5. Build Tree với Stack
        hierarchy_version = {}
        # Stack chứa tuple: (elem_id, level)
        stack = [(self.root_id, self.LEVEL_MAP["document"])]

        for type_, content in tokens:
            if type_ == "itemize_end":
                # Kết thúc môi trường itemize: Pop cho đến khi gặp itemize (level 5)
                while stack and stack[-1][1] > self.LEVEL_MAP["itemize"]:
                    stack.pop()
                # Pop luôn cái itemize container ra khỏi stack (để quay về level section)
                if stack and stack[-1][1] == self.LEVEL_MAP["itemize"]:
                    stack.pop()
                continue
            
            level = self.LEVEL_MAP.get(type_, 7)
            
            # Đăng ký element & lấy ID (có dedup)
            elem_id = self._get_or_create_id(type_, content)
            
            
            while stack and stack[-1][1] >= level:
                stack.pop()
            
            if not stack:
                # Fallback an toàn: nếu stack rỗng (lỗi cấu trúc), gắn vào root
                parent_id = self.root_id
                stack.append((self.root_id, 0))
            else:
                parent_id = stack[-1][0]

            # Ghi nhận quan hệ
            hierarchy_version[elem_id] = parent_id
            
            
            if type_ not in ["sentence", "equation", "figure", "table"]:
                stack.append((elem_id, level))

        # Lưu version (Chỉ hiện thị số sau v)
        v_key = version_id.split('v')[-1]
        self.hierarchy[v_key] = hierarchy_version

    def build_hierarchy_json(self):
        return {
            "elements": self.elements,
            "hierarchy": self.hierarchy
        }
    
    def get_stats(self):
        """
        Tính toán các chỉ số thống kê của cấu trúc cây.
        """
        if not self.hierarchy:
            return {
                "avg_depth": 0,
                "max_depth": 0,
                "avg_branching": 0,
                "total_elements": len(self.elements)
            }

        # Lấy version mới nhất để thống kê
        last_ver = sorted(self.hierarchy.keys())[-1]
        tree_map = self.hierarchy[last_ver] # {child_id: parent_id}
        
        # 1. Build map ngược: {parent_id: [children]}
        children_map = defaultdict(list)
        for child, parent in tree_map.items():
            children_map[parent].append(child)
            
        # 2. Tính Branching Factor (Số con trung bình của các node cha)
        total_children = 0
        parent_nodes_count = 0
        for parent, children in children_map.items():
            total_children += len(children)
            parent_nodes_count += 1
            
        avg_branching = total_children / parent_nodes_count if parent_nodes_count > 0 else 0
        
        # 3. Tính Depth (Độ sâu của từng node so với root)
        depths = []
        for node in tree_map.keys():
            d = 0
            curr = node
            path_visited = set()
            # Leo ngược lên root
            while curr != self.root_id and curr in tree_map:
                if curr in path_visited:
                    break

                path_visited.add(curr)
                curr = tree_map[curr]
                d += 1
                if d > 100: 
                    break

            depths.append(d)
            
        if depths:
            avg_depth = sum(depths) / len(depths)
            max_depth = max(depths)
        else:
            avg_depth = 0
            max_depth = 0
        
        return {
            "avg_depth": round(avg_depth, 2),
            "max_depth": max_depth,
            "avg_branching": round(avg_branching, 2),
            "total_elements": len(self.elements)
        }