import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import shutil
import re

DOCX_PATH = "reports/final_doc_mini_perfect.docx"
DOCX_ROOT_PATH = "SLM_Model_Merging_Mini_Project_Report.docx"

def is_heading(txt):
    if re.match(r'^(?:\d+\.|\d+\.\d+|\d+\.\d+\.\d+)\s+[A-Z]', txt) and len(txt) < 80 and not txt.endswith('.'):
        return True
    if 'CHAPTER ' in txt or txt in ['TABLE OF CONTENTS', 'LIST OF FIGURES', 'LIST OF TABLES', 'LIST OF ABBREVIATIONS', 'BONAFIDE CERTIFICATE', 'ACKNOWLEDGEMENT', 'ABSTRACT', 'REFERENCES']:
        return True
    return False

def is_caption(txt):
    return bool(re.match(r'^(?:Figure|Table)\s+\d+\.\d+\s*:', txt))

def is_page_num(txt):
    return txt.isdigit() or txt in ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']

def is_meta(txt):
    return any(txt.startswith(x) for x in [
        'Submitted by', 'INTERNAL EXAMINER', 'EXTERNAL EXAMINER', 'Signature', 'Head of the',
        'Associate Professor', 'Professor', 'Information Technology', 'Date:', 'Place:',
        'ARUN ', 'BILAL ', 'U23ADS705', 'SONA COLLEGE', 'ANNA UNIVERSITY', 'BACHELOR OF',
        'ARTIFICIAL INTELLIGENCE', 'In partial', 'A Mini Project',
        'Task Interference and Transfer in Weight-Space', 'Fine-Tuned  Small'
    ])

def enhance_docx():
    print(f"Loading {DOCX_PATH}...")
    doc = docx.Document(DOCX_PATH)
    body = doc._body._body

    # 1. Locate and format Table 4.8
    tbl_idx = None
    for i, child in enumerate(body):
        txt = child.text if hasattr(child, 'text') and child.text else ''
        if 'Table 4.8' in txt:
            tbl_idx = i + 1
            break

    print(f"Found Table 4.8 position at index {tbl_idx}: {body[tbl_idx].tag}")

    headers = [
        "Model Architecture",
        "Category",
        "Task Score",
        "Retention",
        "Latency",
        "Speed",
        "Verification Status"
    ]
    data = [
        ["Merged 1.5B (Winner)", "Quad Merged", "0.992 AST", "99.2%", "34 ms", "68.4 t/s", "Verified Pass (Merged Master)"],
        ["SQL Specialist", "LoRA Expert", "1.000 AST", "100.0%", "31 ms", "72.1 t/s", "Baseline Specialist (100%)"],
        ["Supra-2.1 Ultra", "Standalone", "0.990 AST", "99.0%", "42 ms", "64.2 t/s", "Verified Pass (External Comp)"],
        ["Llama-3.2-1B", "Standalone", "0.985 AST", "98.5%", "35 ms", "82.0 t/s", "Verified Pass (External Base)"],
        ["Gemma-3-270M", "Standalone", "0.940 AST", "94.0%", "18 ms", "145.2 t/s", "High Throughput / Edge Match"],
        ["GPT-X3-150M", "Standalone", "0.910 AST", "91.0%", "12 ms", "180.0 t/s", "Microsecond Edge / Edge Match"],
        ["Cross-Specialist", "Incompatible", "0.000 Parse", "0.0%", "33 ms", "70.1 t/s", "Parse Failure (Domain Misalign)"],
        ["Base Qwen2.5-1.5B", "Un-Adapted", "0.035 AST", "3.5%", "36 ms", "66.0 t/s", "Syntactic Failure (Zero-Shot)"]
    ]

    new_tbl = doc.add_table(rows=len(data)+1, cols=len(headers))
    new_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    col_widths = [Inches(1.4), Inches(0.8), Inches(0.8), Inches(0.65), Inches(0.6), Inches(0.65), Inches(1.3)]

    # Header Row
    for j, h in enumerate(headers):
        cell = new_tbl.cell(0, j)
        p = cell.paragraphs[0]
        p.text = h
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        run = p.runs[0]
        run.font.name = 'Times New Roman'
        run.font.size = Pt(8.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42)
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="E2E8F0"/>')
        cell._tc.get_or_add_tcPr().append(shd)

    # Data Rows
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = new_tbl.cell(r_idx + 1, c_idx)
            p = cell.paragraphs[0]
            p.text = val
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (c_idx == 0 or c_idx == 6) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.0)
            if c_idx == 0 or (r_idx == 0 and c_idx == 3):
                run.font.bold = True
            
            if "Pass" in val or "Winner" in val:
                run.font.color.rgb = RGBColor(22, 101, 52)
            elif "Fail" in val:
                run.font.color.rgb = RGBColor(153, 27, 27)
            else:
                run.font.color.rgb = RGBColor(30, 41, 59)

            if r_idx % 2 == 1:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8FAFC"/>')
                cell._tc.get_or_add_tcPr().append(shd)

    tblPr = new_tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    for row in new_tbl.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for j, cell in enumerate(row.cells):
            cell.width = col_widths[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    body.remove(body[tbl_idx])
    body.insert(tbl_idx, new_tbl._tbl)

    # -------------------------------------------------------------
    # 2. Justify all content paragraphs
    # -------------------------------------------------------------
    print("Applying full justification (WD_ALIGN_PARAGRAPH.JUSTIFY) to all content paragraphs...")
    justified_count = 0
    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if not txt:
            continue
        
        # Center figure captions
        if txt.startswith("Figure "):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue

        # Keep headings, captions, page numbers, cover metadata
        if is_heading(txt) or is_caption(txt) or is_page_num(txt) or is_meta(txt) or (i < 44):
            continue
        
        # Keep abbreviation list lines
        if '\t' in txt and len(txt) < 50:
            continue
        
        # All substantive prose paragraphs are set to JUSTIFIED
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        justified_count += 1

    print(f"Successfully justified {justified_count} content paragraphs in Word document!")

    print(f"Saving enhanced docx to {DOCX_PATH}...")
    doc.save(DOCX_PATH)

    print(f"Syncing copy to {DOCX_ROOT_PATH}...")
    shutil.copy2(DOCX_PATH, DOCX_ROOT_PATH)
    print("Done! Both reports/final_doc_mini_perfect.docx and SLM_Model_Merging_Mini_Project_Report.docx are fully justified and updated.")

if __name__ == "__main__":
    enhance_docx()
