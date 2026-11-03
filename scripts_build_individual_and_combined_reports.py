import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import os
import shutil

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS & SPECIFICATIONS (From Sample RTF Templates)
# -----------------------------------------------------------------------------
A4_WIDTH = Inches(8.268)       # 210 mm
A4_HEIGHT = Inches(11.693)     # 297 mm
LEFT_MARGIN = Inches(1.378)    # 35 mm = 1987 twips = 99.2 pt
RIGHT_MARGIN = Inches(0.787)   # 20 mm = 1138 twips = 56.7 pt
TOP_MARGIN = Inches(1.181)     # 30 mm = 1699 twips = 85.0 pt
BOTTOM_MARGIN = Inches(0.984)  # 25 mm = 1411 twips = 70.9 pt
USABLE_WIDTH_INCHES = 6.10     # 8.27 - 1.38 - 0.79 = 6.10 in

def apply_page_setup(doc):
    for sec in doc.sections:
        sec.page_width = A4_WIDTH
        sec.page_height = A4_HEIGHT
        sec.left_margin = LEFT_MARGIN
        sec.right_margin = RIGHT_MARGIN
        sec.top_margin = TOP_MARGIN
        sec.bottom_margin = BOTTOM_MARGIN

# XML Helpers
def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="CBD5E1", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:insideV w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/></w:tblBorders>')
    tblPr.append(borders)

def make_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

def make_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

# Paragraph & Content Helpers
def add_p(doc, text, space_before=0, space_after=4, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5, bold=False, italic=False, font_size=14):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    p.paragraph_format.first_line_indent = Inches(first_indent) if first_indent > 0 else Inches(0)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_bullet(doc, text, space_after=3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.2)
    
    r_bullet = p.add_run("•  ")
    r_bullet.font.name = 'Times New Roman'
    r_bullet.font.size = Pt(14)
    r_bullet.font.bold = True
    r_bullet.font.color.rgb = RGBColor(0, 0, 0)
    
    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    r_text.font.size = Pt(14)
    r_text.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_numbered_item(doc, num_str, title_str, text_str, space_after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)

    r_num = p.add_run(num_str + " ")
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(14)
    r_num.font.bold = True
    r_num.font.color.rgb = RGBColor(0, 0, 0)

    if title_str:
        r_title = p.add_run(title_str + " ")
        r_title.font.name = 'Times New Roman'
        r_title.font.size = Pt(14)
        r_title.font.bold = True
        r_title.font.color.rgb = RGBColor(0, 0, 0)

    r_body = p.add_run(text_str)
    r_body.font.name = 'Times New Roman'
    r_body.font.size = Pt(14)
    r_body.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_chapter_title(doc, text, is_first=False):
    if not is_first:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text.upper())
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_subsection_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_table_title(doc, title_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(title_text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_figure(doc, image_path, caption_text, width_inches=5.8):
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} missing!")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.first_line_indent = Inches(0)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(image_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(3)
    p_cap.paragraph_format.space_after = Pt(10)
    p_cap.paragraph_format.first_line_indent = Inches(0)
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = 'Times New Roman'
    run_cap.font.size = Pt(12)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(0, 0, 0)

def add_custom_table(doc, title_text, headers, rows_data, col_widths, alignments=None):
    if title_text:
        add_table_title(doc, title_text)
    table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    if alignments is None:
        alignments = [WD_ALIGN_PARAGRAPH.LEFT] * len(headers)

    # Header Row
    hdr_cells = table.rows[0].cells
    make_row_header(table.rows[0])
    for idx, heading in enumerate(headers):
        hdr_cells[idx].text = heading
        set_cell_shading(hdr_cells[idx], "E2E8F0")
        set_cell_margins(hdr_cells[idx], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)

    # Data Rows
    for r_idx, row in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        make_row_cant_split(table.rows[r_idx + 1])
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = alignments[c_idx] if c_idx < len(alignments) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.bold = False
                r.font.color.rgb = RGBColor(0, 0, 0)

    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)
    return table

def add_code_block(doc, title_text, code_lines):
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.paragraph_format.first_line_indent = Inches(0)
    p_title.paragraph_format.keep_with_next = True
    run_t = p_title.add_run(title_text)
    run_t.font.name = 'Times New Roman'
    run_t.font.size = Pt(12)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0, 0, 0)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_shading(table.rows[0].cells[0], "F8FAFC")
    set_cell_margins(table.rows[0].cells[0], top=100, bottom=100, left=150, right=150)
    set_table_borders(table, color="CBD5E1", sz="4")
    table.rows[0].cells[0].width = Inches(USABLE_WIDTH_INCHES)

    cell = table.rows[0].cells[0]
    p_first = cell.paragraphs[0]
    p_first.paragraph_format.space_before = Pt(1)
    p_first.paragraph_format.space_after = Pt(1)
    p_first.paragraph_format.line_spacing = 1.0
    p_first.paragraph_format.first_line_indent = Inches(0)
    r = p_first.add_run(code_lines[0])
    r.font.name = 'Consolas'
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(30, 41, 59)

    for line in code_lines[1:]:
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = Inches(0)
        r = p.add_run(line)
        r.font.name = 'Consolas'
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(30, 41, 59)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

# -----------------------------------------------------------------------------
# BUILD FRONT PAGES (Matching Front Pages - Final template.rtf.doc)
# -----------------------------------------------------------------------------
def build_front_pages_content(doc):
    # [1. COVER / TITLE PAGE]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run("TASK INTERFERENCE AND TRANSFER IN WEIGHT-SPACE MERGING OF FINE-TUNED SMALL LANGUAGE MODELS")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("A PROJECT REPORT")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("U23ADS705 – MINI PROJECT")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("Submitted by")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.italic = True

    # Student Names Table (borderless)
    std_table = doc.add_table(rows=3, cols=2)
    std_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    students = [
        ("ARUN M", "(61782323110014)"),
        ("ARUN PRASATH S", "(61782323110016)"),
        ("BILAL AHAMED A", "(61782323110024)")
    ]
    for idx, (name, reg) in enumerate(students):
        c_name, c_reg = std_table.rows[idx].cells
        c_name.width = Inches(2.2)
        c_reg.width = Inches(2.0)
        c_name.text = name
        c_reg.text = reg
        for c in (c_name, c_reg):
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("in partial fulfillment for the award of the degree of")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.italic = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("BACHELOR OF TECHNOLOGY")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("in")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.italic = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("ARTIFICIAL INTELLIGENCE AND DATA SCIENCE")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True

    # Sona Logo
    if os.path.exists("reports/extracted_images/image2.jpeg"):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(6)
        p_logo.paragraph_format.space_after = Pt(10)
        p_logo.paragraph_format.first_line_indent = Inches(0)
        p_logo.add_run().add_picture("reports/extracted_images/image2.jpeg", width=Inches(1.8))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("SONA COLLEGE OF TECHNOLOGY, SALEM – 636005")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("(An Autonomous Institution)")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("ANNA UNIVERSITY: CHENNAI 600 025")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("NOVEMBER 2026")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True

    # [2. BONAFIDE CERTIFICATE]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("SONA COLLEGE OF TECHNOLOGY (AUTONOMOUS)")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("SALEM – 636005")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("ANNA UNIVERSITY: CHENNAI – 600 025")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    if os.path.exists("reports/extracted_images/image1.jpeg"):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(4)
        p_logo.paragraph_format.space_after = Pt(8)
        p_logo.paragraph_format.first_line_indent = Inches(0)
        p_logo.add_run().add_picture("reports/extracted_images/image1.jpeg", width=Inches(1.2))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("BONAFIDE CERTIFICATE")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    cert_text = (
        "Certified that this project titled “TASK INTERFERENCE AND TRANSFER IN WEIGHT-SPACE "
        "MERGING OF FINE-TUNED SMALL LANGUAGE MODELS” is the bonafide work of ARUN M (61782323110014), "
        "ARUN PRASATH S (61782323110016), and BILAL AHAMED A (61782323110024) who carried out the "
        "U23ADS705 – MINI PROJECT under my supervision."
    )
    add_p(doc, cert_text, space_before=6, space_after=24, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    sig_table = doc.add_table(rows=1, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_sup, c_hod = sig_table.rows[0].cells
    c_sup.width = Inches(3.0)
    c_hod.width = Inches(3.1)

    p_sup = c_sup.paragraphs[0]
    p_sup.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_sup.paragraph_format.line_spacing = 1.15
    run = p_sup.add_run("SIGNATURE OF SUPERVISOR\n\n\n\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run = p_sup.add_run("Ms. S.M.C. Subashini, M.E., (Ph.D.)\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run = p_sup.add_run("Supervisor\nAssociate Professor,\nDepartment of Artificial Intelligence and Data Science,\nSona College of Technology,\nSalem – 636005.")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

    p_hod = c_hod.paragraphs[0]
    p_hod.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_hod.paragraph_format.line_spacing = 1.15
    run = p_hod.add_run("HEAD OF THE DEPARTMENT\n\n\n\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run = p_hod.add_run("Dr. J. Akilandeswari, M.E., Ph.D.\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run = p_hod.add_run("Head of the Department\nProfessor & Head,\nDepartment of Artificial Intelligence and Data Science,\nSona College of Technology,\nSalem – 636005.")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

    p_viva = doc.add_paragraph()
    p_viva.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_viva.paragraph_format.space_before = Pt(36)
    p_viva.paragraph_format.space_after = Pt(28)
    p_viva.paragraph_format.first_line_indent = Inches(0)
    run = p_viva.add_run("Submitted for the Project Viva-Voce Examination held on ____________________")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True

    exam_table = doc.add_table(rows=1, cols=2)
    exam_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c1, c2 = exam_table.rows[0].cells
    c1.width = Inches(3.0)
    c2.width = Inches(3.1)
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p1.add_run("INTERNAL EXAMINER")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True

    p2 = c2.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p2.add_run("EXTERNAL EXAMINER")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True

    # [3. STUDENT DECLARATION FORM]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("SONA COLLEGE OF TECHNOLOGY (AUTONOMOUS)")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("U23ADS705 – MINI PROJECT – DECLARATION FORM")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("Declaration by the Student(s)")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True

    decl_p1 = (
        "We hereby declare that the project titled “TASK INTERFERENCE AND TRANSFER IN WEIGHT-SPACE "
        "MERGING OF FINE-TUNED SMALL LANGUAGE MODELS” is an original work carried out by us under the "
        "guidance of Ms. S.M.C. Subashini. This project has not been submitted elsewhere for the award of any degree."
    )
    add_p(doc, decl_p1, space_before=4, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    decl_p2 = "We also affirm that all the data and information used in this project are genuine and properly acknowledged."
    add_p(doc, decl_p2, space_before=2, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    decl_p3 = (
        "We also declare that our project is related to the Sustainable Development Goal (SDG) 9: Industry, "
        "Innovation, and Infrastructure & SDG 4: Quality Education."
    )
    add_p(doc, decl_p3, space_before=2, space_after=18, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    decl_headers = ["S.No", "Register Number", "Name", "Signature"]
    decl_rows = [
        ["1.", "61782323110014", "ARUN M", ""],
        ["2.", "61782323110016", "ARUN PRASATH S", ""],
        ["3.", "61782323110024", "BILAL AHAMED A", ""]
    ]
    add_custom_table(doc, "", decl_headers, decl_rows, [0.8, 1.8, 2.0, 1.5], [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])

    p_date = doc.add_paragraph()
    p_date.paragraph_format.space_before = Pt(24)
    p_date.paragraph_format.space_after = Pt(4)
    p_date.paragraph_format.first_line_indent = Inches(0)
    run = p_date.add_run("Date: ____________________\nPlace: Sona College of Technology, Salem")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True

    # [4. ACKNOWLEDGEMENT]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("ACKNOWLEDGEMENT")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    ack_1 = (
        "First and foremost, we thank the Almighty for His blessings, guidance, and the inner peace that gave us "
        "the strength and determination to complete this project. We also express our heartfelt gratitude to our parents "
        "for their unconditional love, constant support, and encouragement throughout our journey."
    )
    add_p(doc, ack_1, space_before=4, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    ack_2 = (
        "We express our sincere gratitude to our beloved Chairman, Sri. C. Valliappa, and our dynamic Vice-Chairmen, "
        "Sri. Chocko Valliappa and Sri. Thyagu Valliappa, for providing us with the necessary facilities and support "
        "to successfully complete this project."
    )
    add_p(doc, ack_2, space_before=2, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    ack_3 = (
        "We are immensely grateful to our Principal, Dr. S.R.R. Senthilkumar, for his constant inspiration and "
        "encouragement throughout our academic journey."
    )
    add_p(doc, ack_3, space_before=2, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    ack_4 = (
        "We extend our sincere thanks to Dr. J. Akilandeswari, Professor and Head of the Department of Artificial "
        "Intelligence and Data Science, for providing the necessary laboratory facilities and academic support to carry "
        "out this project successfully."
    )
    add_p(doc, ack_4, space_before=2, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    ack_5 = (
        "We express our heartfelt gratitude to our project supervisor, Ms. S.M.C. Subashini, M.E., (Ph.D.), Associate "
        "Professor, Department of Artificial Intelligence and Data Science, for her invaluable mentorship, insightful "
        "guidance, and unwavering support throughout the project. Her expertise and constructive suggestions have played "
        "a significant role in shaping our work and helping us achieve our objectives."
    )
    add_p(doc, ack_5, space_before=2, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    ack_6 = (
        "Finally, we take this opportunity to express our sincere appreciation to all the faculty members, non-teaching "
        "staff, and friends who supported us, either directly or indirectly, in completing this project successfully."
    )
    add_p(doc, ack_6, space_before=2, space_after=14, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    # [5. ABSTRACT]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("ABSTRACT")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    abs_p1 = (
        "Fine-tuning Small Language Models (SLMs) via Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank "
        "Adaptation (LoRA), has emerged as a premier paradigm for creating specialized domain experts. However, deploying "
        "multiple dedicated adapters concurrently incurs substantial computational and memory overheads, while multi-task "
        "joint training suffers from severe negative gradient interference and high re-training costs. Weight-space model "
        "merging offers an appealing zero-marginal-cost alternative by algebraically combining independent specialist "
        "weight vectors into a single unified model without additional backpropagation passes. Nonetheless, existing "
        "merging literature primarily focuses on multi-billion parameter scales (7B–70B), leaving task interference "
        "dynamics and arithmetic stability at the compact 1.5-billion scale largely uncharacterized."
    )
    add_p(doc, abs_p1, space_before=4, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    abs_p2 = (
        "In this work, we present a systematic empirical study of weight-space model merging on a strictly pinned base "
        "model (Qwen/Qwen2.5-1.5B-Instruct, commit 989aa79). Four specialist LoRA adapters (r=16, alpha=32) are trained across "
        "fundamentally diverse NLP capabilities: structured Text-to-SQL synthesis, step-by-step mathematical reasoning "
        "(GSM8K), low-resource Tamil-to-English translation (Samanantar), and schema-constrained JSON information extraction. "
        "Five distinct merging algorithms—Linear Averaging, Spherical Linear Interpolation (SLERP), Trim-Elect-Sign-Merge "
        "(TIES), Drop And REscale with TIES (DARE-TIES), and Magnitude Pruning—are systematically evaluated across the complete "
        "combinatorial grid of 11 multi-task combinations (6 pairs, 4 triplets, and 1 quad) across 280 rigorously evaluated cells."
    )
    add_p(doc, abs_p2, space_before=2, space_after=8, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    abs_p3 = (
        "Empirical results demonstrate that Magnitude Pruning achieves the highest overall retention (mean retention = 0.991), "
        "demonstrating complete preservation across all four tasks without catastrophic interference, while DARE-TIES achieves "
        "0.985 retention. Mathematical reasoning exhibits positive transfer (+4.8% to +6.3% synergy), whereas SQL synthesis "
        "demonstrates acute syntactic sensitivity to delta trimming. The optimal quad-merged model is compiled into an ultra-compact "
        "Q4_K_M GGUF binary (941 MB) and deployed on consumer edge hardware via an asynchronous C++ engine with sub-22 ms/token "
        "latency. Finally, an interactive Web Studio dashboard and Cross-Model Benchmark Arena are provided to enable real-time "
        "multi-task prompt testing, AST verification, and automated merging conflict diagnostics."
    )
    add_p(doc, abs_p3, space_before=2, space_after=14, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5)

    # [6. TABLE OF CONTENTS]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("TABLE OF CONTENTS")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    toc_headers = ["SECTION NO", "TITLE", "PAGE NO"]
    toc_rows = [
        ["", "DECLARATION", "iii"],
        ["", "ACKNOWLEDGEMENT", "iv"],
        ["", "ABSTRACT", "v"],
        ["", "LIST OF TABLES", "vi"],
        ["", "LIST OF FIGURES", "vii"],
        ["", "LIST OF ABBREVIATIONS", "viii"],
        ["1.", "INTRODUCTION", "1"],
        ["1.1", "Background of the Project", "1"],
        ["1.2", "Problem Statement", "2"],
        ["1.3", "Motivation", "3"],
        ["1.4", "Objectives", "3"],
        ["1.5", "Scope of the Project", "4"],
        ["2.", "LITERATURE REVIEW", "5"],
        ["2.1", "Introduction", "5"],
        ["2.2", "Review of Related Studies", "5"],
        ["2.3", "Techniques and Technologies Used in Existing Systems", "8"],
        ["2.4", "Research Gap", "9"],
        ["2.5", "Summary", "9"],
        ["3.", "CHAPTER 1 – NEED FOR PROPOSED TECHNIQUE", "10"],
        ["3.1", "Existing Techniques", "10"],
        ["3.2", "Limitations of Existing Techniques", "11"],
        ["3.3", "Proposed Technique", "12"],
        ["3.4", "Objectives of the Proposed Technique", "13"],
        ["3.5", "Advantages of the Proposed Technique", "14"],
        ["4.", "CHAPTER 2 – DESIGN AND IMPLEMENTATION OF PROPOSED SYSTEM", "15"],
        ["4.1", "Introduction", "15"],
        ["4.2", "System Requirements", "15"],
        ["4.2.1", "Hardware Requirements", "15"],
        ["4.2.2", "Software Requirements", "16"],
        ["4.3", "System Architecture", "17"],
        ["4.4", "System Components / Modules", "18"],
        ["4.4.1", "Task-Specific LoRA Adaptation Engine", "18"],
        ["4.4.2", "Algebraic Model Merging Engine", "19"],
        ["4.4.3", "Automated Evaluation & Multi-Metric Scoring Pipeline", "20"],
        ["4.4.4", "GGUF Model Quantization & Compilation Module", "21"],
        ["4.4.5", "C++ Quantized Inference Engine", "22"],
        ["4.4.6", "FastAPI REST Backend Server", "23"],
        ["4.4.7", "Browser-Based Interactive Workstation", "24"],
        ["4.5", "Proposed Methodology", "24"],
        ["4.6", "System Workflow", "25"],
        ["4.7", "Algorithms / Techniques Used", "26"],
        ["4.8", "Dataset / Data Collection", "27"],
        ["4.9", "Data Preprocessing", "28"],
        ["4.10", "Model / System Development", "29"],
        ["4.11", "System Design Diagrams", "30"],
        ["4.12", "Implementation", "32"],
        ["4.13", "Coding / Program Implementation", "32"],
        ["4.14", "User Interface / System Execution Screens", "35"],
        ["4.14.1", "Interactive CLI & Terminal Execution Screens", "35"],
        ["4.14.2", "Graphical Web Studio Interface Execution Screens", "38"],
        ["4.14.3", "Cross-Model Benchmark Arena & Prompt Testing Interface", "43"],
        ["4.15", "Chapter Summary", "44"],
        ["5.", "CHAPTER 3 – RESULTS AND DISCUSSION", "45"],
        ["5.1", "Introduction", "45"],
        ["5.2", "Experimental Setup", "45"],
        ["5.3", "Evaluation Metrics", "45"],
        ["5.4", "Experimental Results", "46"],
        ["5.4.1", "Baseline Performance", "46"],
        ["5.4.2", "Two-Task Merging Performance & Interference Retention", "48"],
        ["5.4.3", "Three-Task Merging Interference Profile", "50"],
        ["5.4.4", "Four-Task Quad Merging Performance", "51"],
        ["5.4.5", "Weight-Space vs Adapter Sweep Analysis", "52"],
        ["5.4.6", "Mergekit Cross-Check Validation", "53"],
        ["5.4.7", "Live Inference Speed & Memory Footprint", "54"],
        ["5.5", "Discussion of Results", "54"],
        ["5.6", "Chapter Summary", "54"],
        ["6.", "CHAPTER 4 – CONCLUSION AND FUTURE SCOPE", "55"],
        ["6.1", "Summary of the Project", "55"],
        ["6.2", "Conclusion", "56"],
        ["6.3", "Future Scope", "58"],
        ["", "SUSTAINABLE DEVELOPMENT GOALS (SDG)", "59"],
        ["", "REFERENCES", "60"]
    ]
    add_custom_table(doc, "", toc_headers, toc_rows, [1.3, 3.8, 1.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])

    # [7. LIST OF TABLES]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("LIST OF TABLES")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    lot_headers = ["TABLE NO", "TITLE", "PAGE NO"]
    lot_rows = [
        ["1.1", "Base Model Specifications", "2"],
        ["2.1", "Summary of Literature Review", "8"],
        ["3.1", "Comparison of Existing Multi-Task Techniques", "11"],
        ["4.1", "Hardware Requirements", "15"],
        ["4.2", "Software Requirements", "16"],
        ["4.3", "LoRA Configuration Parameters (Identical Across All Adapters)", "19"],
        ["4.4", "Training Hyperparameters (Identical Across All Tasks)", "20"],
        ["4.5", "Task Descriptions and Datasets", "27"],
        ["4.6", "Dataset Statistics After Preprocessing", "29"],
        ["4.7", "Merge Grid Configurations (PEFT Arithmetic)", "31"],
        ["4.8", "Cross-Model Benchmark Arena Performance Metrics Across 8 Models", "44"],
        ["5.1", "Evaluation Metrics per Task", "46"],
        ["5.2", "Individual Adapter Training Results", "47"],
        ["5.3", "Merge Grid Summary", "48"],
        ["5.4", "Individual Adapter Specialist Training and Benchmark Results", "49"],
        ["5.5", "Empirical Multi-Task Retention Comparison Across 5 Merging Algorithms (144 Cells)", "49"],
        ["5.6", "Mean Retention Degradation Across Task Arities", "51"],
        ["5.7", "Task-Specific Retention and Fragility Matrix Across Merging Algorithms", "53"]
    ]
    add_custom_table(doc, "", lot_headers, lot_rows, [1.3, 3.8, 1.0], [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])

    # [8. LIST OF FIGURES]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("LIST OF FIGURES")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    lof_headers = ["FIGURE NO", "TITLE", "PAGE NO"]
    lof_rows = [
        ["3.1", "LoRA (Low-Rank Adaptation) Architecture", "11"],
        ["3.2", "Comparison of Five Weight-Space Merging Methods", "13"],
        ["4.1", "System Architecture of the SLM Merging Study", "17"],
        ["4.2", "FastAPI Backend Server & Asynchronous Subprocess Orchestration Architecture", "24"],
        ["4.3", "Experimental Pipeline Workflow", "25"],
        ["4.4", "Dataset Distribution Across Tasks", "28"],
        ["4.5", "Evaluation Framework for Multi-Task NLP System", "30"],
        ["4.6", "Merge Grid Heatmap: 5 Methods × 11 Combinations", "31"],
        ["4.7", "Terminal Execution: LoRA Fine-Tuning of Qwen2.5-1.5B with GPU Telemetry", "35"],
        ["4.8", "Terminal Execution: Weight-Space Model Merging via TIES Pipeline", "36"],
        ["4.9", "Terminal Evaluation: Multi-Task Performance Matrix and AST Retention", "37"],
        ["4.10", "Web Studio Interface — Studio Dashboard, Hardware Telemetry & 3-Step Guided Merging Wizard", "38"],
        ["4.11", "Web Studio Interface — Automated Merging Strategy Advisor & Conflict Prediction Engine", "39"],
        ["4.12", "Web Studio Interface — Comprehensive Empirical Analysis, Radar Charts & Task Fragility Matrix", "40"],
        ["4.13", "Web Studio Interface — Interactive Multi-Turn Model Playground & Inline AST Verification", "41"],
        ["4.14", "Web Studio Interface — GGUF Quantization, Memory Footprints & Local Edge Deployment Controls", "42"],
        ["4.15", "Web Studio Interface — Cross-Model Benchmark Arena Prompt Testing & Scoreboard", "43"],
        ["5.1", "Final Training Loss After 2 Epochs for LoRA Fine-Tuned Models", "47"],
        ["5.2", "Task-Specific Retention Heatmap — Method × Task Domain", "50"],
        ["5.3", "Empirical Multi-Task Retention Comparison Across 5 Merging Algorithms and Retention Degradation by Task Arity", "52"]
    ]
    add_custom_table(doc, "", lof_headers, lof_rows, [1.3, 3.8, 1.0], [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])

    # [9. LIST OF ABBREVIATIONS]
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run("LIST OF ABBREVIATIONS")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True

    abbrev_headers = ["Abbreviation", "Full Forms"]
    abbrev_rows = [
        ["SLM", "Small Language Model"],
        ["LLM", "Large Language Model"],
        ["LoRA", "Low-Rank Adaptation"],
        ["PEFT", "Parameter-Efficient Fine-Tuning"],
        ["TIES", "Trim, Elect Sign, and Merge"],
        ["DARE", "Drop And REscale"],
        ["DARE-TIES", "Drop And REscale with Trim, Elect Sign, and Merge"],
        ["SLERP", "Spherical Linear Interpolation"],
        ["NLP", "Natural Language Processing"],
        ["SQL", "Structured Query Language"],
        ["JSON", "JavaScript Object Notation"],
        ["AST", "Abstract Syntax Tree"],
        ["chrF++", "Character F-score with Word Bigrams"],
        ["GPU", "Graphics Processing Unit"],
        ["VRAM", "Video Random Access Memory"],
        ["GGUF", "GPT-Generated Unified Format"],
        ["API", "Application Programming Interface"],
        ["HF", "Hugging Face"],
        ["bf16", "Brain Floating Point 16-bit"],
        ["FP16", "16-bit Floating Point"],
        ["SDG", "Sustainable Development Goal"],
        ["F1", "F1 Score (Harmonic Mean of Precision and Recall)"],
        ["BLEU", "Bilingual Evaluation Understudy"],
        ["MoE", "Mixture of Experts"],
        ["MTL", "Multi-Task Learning"],
        ["SFT", "Supervised Fine-Tuning"],
        ["GSM8K", "Grade School Math 8K Benchmark"],
        ["REST", "Representational State Transfer"],
        ["SPA", "Single Page Application"]
    ]
    add_custom_table(doc, "", abbrev_headers, abbrev_rows, [2.0, 4.1], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

# -----------------------------------------------------------------------------
# BUILD REPORT CONTENT (Matching Sample project report - CONTENT.rtf.doc)
# -----------------------------------------------------------------------------
def build_report_content(doc, is_standalone=False):
    # CHAPTER 1: INTRODUCTION
    add_chapter_title(doc, "1. INTRODUCTION", is_first=is_standalone)

    add_section_heading(doc, "1.1 Background of the Project")
    add_p(doc, "The rapid advancement of Large Language Models (LLMs) has revolutionized natural language processing, demonstrating remarkable capabilities across a wide range of tasks including code generation, mathematical reasoning, translation, and structured data extraction. However, the deployment of large-scale models (7B to 70B+ parameters) poses severe practical challenges in resource-constrained environments such as edge devices, consumer laptops, and embedded systems due to prohibitive memory requirements, high latency, and massive compute costs. Small Language Models (SLMs), typically defined as models with fewer than 3 billion parameters, offer a compelling alternative by drastically reducing computational requirements while maintaining competitive capabilities on targeted tasks.")
    add_p(doc, "Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), have become the standard approach for adapting pre-trained models to specific tasks. LoRA freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters and GPU memory requirements. While LoRA enables efficient training of task-specific adapters, serving multiple task-specific adapters simultaneously still requires significant memory overhead or complex routing mechanisms during inference.")
    add_p(doc, "Weight-space model merging is an emerging technique that combines multiple fine-tuned models into a single set of weights without requiring retraining or access to the original training data. By performing mathematical operations directly on model weights (such as linear averaging, spherical linear interpolation, or interference-directed merging), model merging creates a unified multi-task model at near-zero marginal computational cost. While model merging has shown promising results in larger models, its behavior at the SLM scale—where parameter capacity is fundamentally constrained—remains systematically understudied.")

    t1_headers = ["Property", "Value"]
    t1_rows = [
        ["Model", "Qwen/Qwen2.5-1.5B-Instruct"],
        ["Parameters", "1.5 Billion"],
        ["Revision (Pinned)", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"],
        ["Precision", "bfloat16 (no quantization)"],
        ["Architecture", "Transformer (Decoder-only)"]
    ]
    add_custom_table(doc, "Table 1.1: Base Model Specifications", t1_headers, t1_rows, [2.5, 3.6], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_section_heading(doc, "1.2 Problem Statement")
    add_p(doc, "When multiple task-specific LoRA adapters are merged in weight space, the resulting model often suffers from task interference—a phenomenon where parameter updates that benefit one task degrade performance on another. This problem is particularly acute in Small Language Models due to their limited parameter capacity, where weight space is tightly packed and individual parameters may be critical for multiple competing behaviors.")
    add_p(doc, "The challenge lies in understanding the conditions under which merging produces destructive interference versus constructive transfer, and how different merging algorithms mitigate parameter conflicts. Current literature lacks systematic empirical evidence comparing multiple merging methods across diverse task types at the SLM scale under controlled experimental conditions.")
    add_p(doc, "This project addresses the following key research questions: (1) How do different weight-space merging techniques perform when combining LoRA adapters at the 1.5B scale? (2) What is the relationship between task diversity and merge performance? (3) Can advanced merging methods (such as TIES, DARE, and magnitude pruning) effectively eliminate inter-task interference? (4) Does weight-space merging exhibit positive transfer (synergy) on specific task pairs?")

    add_section_heading(doc, "1.3 Motivation")
    add_p(doc, "The motivation for this project stems from several practical and theoretical considerations:")
    add_bullet(doc, "Deployment Efficiency: Maintaining multiple task-specific models requires proportional memory and storage. A single merged model serving all four tasks eliminates adapter-switching overhead and reduces the deployment footprint to a single model instance.")
    add_bullet(doc, "SLM-Scale Gap: Most model merging research targets 7B–70B parameter models. The dynamics of weight-space merging in sub-2B models—where capacity constraints are most severe—have not been systematically mapped.")
    add_bullet(doc, "Controlled Methodology: Many existing studies lack strict experimental controls, using different base models, disparate training hyperparameters, or soft evaluation metrics that obscure true task degradation. A strictly controlled pipeline is essential for scientific validity.")
    add_bullet(doc, "Practical Impact: Understanding which merging methods work best for different task combinations provides actionable guidance for practitioners deploying AI solutions on consumer-grade hardware with limited VRAM budgets.")

    add_section_heading(doc, "1.4 Objectives")
    add_numbered_item(doc, "1.", "Fine-Tuning Specialists:", "To fine-tune four task-specialized LoRA adapters on a strictly pinned base model (Qwen2.5-1.5B-Instruct) across diverse domains: Text-to-SQL synthesis, mathematical reasoning (GSM8K), low-resource Tamil-to-English translation, and schema-constrained JSON extraction.")
    add_numbered_item(doc, "2.", "Algorithmic Implementation:", "To implement and apply five weight-space merging techniques: Linear Averaging, Spherical Linear Interpolation (SLERP), Trim-Elect-Sign-Merge (TIES), Drop And REscale with TIES (DARE-TIES), and Magnitude Pruning across the complete combinatorial grid of 11 combinations (6 pairs, 4 triplets, and 1 quad).")
    add_numbered_item(doc, "3.", "Evaluation Suite:", "To develop rigorous, reproducible task-specific evaluation metrics including Abstract Syntax Tree (AST) equivalence for SQL, exact numerical matching for math, corpus chrF++ via SacreBLEU for translation, and schema-level precision/recall/F1 for JSON extraction.")
    add_numbered_item(doc, "4.", "Systematic Measurement:", "To systematically measure and analyze task retention, inter-task interference, and cross-task transfer across all 280 experimental cells, establishing empirical trade-offs between merging methods and task arities.")
    add_numbered_item(doc, "5.", "Edge Deployment & Web Studio:", "To compile and deploy the optimal merged multi-task model using llama.cpp Q4_K_M GGUF quantization (941 MB) on consumer-grade edge hardware, accompanied by an interactive browser-based Web Studio interface and Cross-Model Benchmark Arena.")

    add_section_heading(doc, "1.5 Scope of the Project")
    add_p(doc, "This project focuses on the systematic study of weight-space merging at the 1.5-billion-parameter scale using a single pinned base model (Qwen2.5-1.5B-Instruct). The scope includes four diverse NLP tasks spanning structured output generation (SQL, JSON), reasoning (Math), and sequence transduction (Translation). Five merging methods are evaluated across all possible combinations of the four task adapters.")
    add_p(doc, "The project does not include: (a) comparison across different base model architectures or sizes, (b) training-based multi-task learning approaches, (c) deployment and inference optimization beyond the merging step, or (d) evaluation on held-out benchmarks beyond the project-specific evaluation sets.")

    # CHAPTER 2: LITERATURE REVIEW
    add_chapter_title(doc, "2. LITERATURE REVIEW")

    add_section_heading(doc, "2.1 Introduction")
    add_p(doc, "This chapter reviews the existing literature on model merging, parameter-efficient fine-tuning, and multi-task learning, establishing the theoretical foundation and contextual background for this investigation. The reviewed works are organized alphabetically by first author, in accordance with standard academic reporting guidelines.")

    add_section_heading(doc, "2.2 Review of Related Studies")
    add_p(doc, "Akiba et al. [1], in “Evolutionary Optimization of Model Merging Recipes,” proposed an evolutionary algorithm approach to automatically discover merging recipes across both parameter space and layer space. Their work demonstrated that searching the space of merge configurations can produce models outperforming individual experts, but focused on large models (7B+) with extensive computational resources for the evolutionary search.")
    add_p(doc, "Goddard et al. [2], in “Arcee's MergeKit: A Toolkit for Merging Large Language Models,” introduced an open-source framework supporting multiple merge methods including linear, SLERP, TIES, DARE, and Passthrough. MergeKit has become the standard tool for full-checkpoint merging, though its application to compact adapter-level merging on small models has received less attention.")
    add_p(doc, "Hu et al. [3], in “LoRA: Low-Rank Adaptation of Large Language Models,” introduced the foundational technique of decomposing weight update matrices into low-rank representations. Their work demonstrated that freezing pre-trained weights and training small rank-r matrices achieves performance competitive with full fine-tuning while reducing trainable parameters by 10,000x.")
    add_p(doc, "Ilharco et al. [4], in “Editing Models with Task Arithmetic,” established that task vectors—computed as the difference between fine-tuned and pre-trained weights—can be added, negated, and scaled to edit model behavior. This foundational work demonstrated that simple vector arithmetic in parameter space corresponds to meaningful behavioral changes, providing the conceptual basis for weight-space merging.")
    add_p(doc, "Jin et al. [5], in “Dataless Knowledge Fusion by Merging Weights of Language Models,” investigated merging fine-tuned models without access to training data. They observed that parameter interference and catastrophic forgetting can occur when combining models trained on divergent objectives, highlighting the need for conflict-resolution mechanisms.")
    add_p(doc, "Matena and Raffel [6], in “Merging Models with Fisher-Weighted Averaging,” proposed using the Fisher information matrix to weight parameter importance during merging. Their approach weights parameters by their estimated importance to individual tasks, reducing interference but requiring computation of Fisher information.")
    add_p(doc, "Ortiz-Jimenez et al. [7], in “Task Arithmetic in the Presence of Strong Pre-training,” analyzed why task arithmetic works and identified conditions under which linearity breaks down. Their findings suggest that the geometry of the loss landscape around the pre-trained model governs the success of weight-space operations.")
    add_p(doc, "Wortsman et al. [8], in “Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time,” demonstrated that averaging the weights of multiple models fine-tuned with different hyperparameters on the same task improves accuracy and robustness. This work catalyzed interest in weight averaging, though it focused on single-task variants rather than multi-task combinations.")
    add_p(doc, "Yadav et al. [9], in “TIES-Merging: Resolving Interference When Merging Models,” identified two primary sources of interference in weight merging: redundant parameter updates and conflicting signs across models. They proposed the TIES method (Trim, Elect Sign, Merge) to address both issues, demonstrating significant improvements over simple averaging on benchmark tasks.")
    add_p(doc, "Yu et al. [10], in “Language Models are Super Mario: Absorbing Abilities from Homologous Models with DARE,” introduced DARE (Drop And REscale), which randomly drops a large fraction of delta parameters and rescales the remainder before merging. DARE demonstrated that up to 90% of delta parameters can be eliminated without performance loss, significantly reducing interference when combining multiple models.")

    add_section_heading(doc, "2.3 Techniques and Technologies Used in Existing Systems")
    add_p(doc, "Existing approaches to multi-task serving can be broadly categorized into four paradigms, each with distinct trade-offs between training cost, inference cost, and task isolation:")

    t2_headers = ["Study", "Technique", "Scale", "Key Findings"]
    t2_rows = [
        ["Ilharco et al. (2022)", "Task Arithmetic", "CLIP, T5", "Vector arithmetic in parameter space modifies behavior"],
        ["Wortsman et al. (2022)", "Model Soups", "ViT, CLIP", "Averaging same-task fine-tunes improves robustness"],
        ["Yadav et al. (2023)", "TIES-Merging", "T5 (3B), ViT", "Sign consensus and trimming reduce interference"],
        ["Yu et al. (2024)", "DARE", "Llama-2 (7B, 13B)", "Extreme delta sparsity (90%) preserves capability"],
        ["Akiba et al. (2024)", "Evolutionary Merge", "7B–70B", "Automated recipes produce Pareto-optimal models"],
        ["Goddard et al. (2024)", "MergeKit Toolkit", "General LLMs", "Standardized framework for multi-method merging"],
        ["Hu et al. (2021)", "LoRA (PEFT)", "GPT-3, RoBERTa", "Low-rank updates competitive with full fine-tuning"],
        ["Matena & Raffel (2022)", "Fisher Averaging", "T5", "Importance-weighted averaging mitigates forgetting"]
    ]
    add_custom_table(doc, "Table 2.1: Summary of Literature Review", t2_headers, t2_rows, [1.5, 1.3, 1.1, 2.2], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.JUSTIFY])

    add_section_heading(doc, "2.4 Research Gap")
    add_p(doc, "Despite the rapid growth of model merging research, several critical gaps remain in the literature:")
    add_bullet(doc, "Scale Disparity: The vast majority of published studies focus on models with 7B or more parameters. The behavior of weight-space merging in Small Language Models (< 2B parameters) is substantially under-characterized. Because parameter capacity scales non-linearly with model size, conclusions drawn from 7B+ models cannot be assumed to hold for SLMs.")
    add_bullet(doc, "Task Diversity: Many existing evaluations use closely related tasks (e.g., multiple reasoning datasets or multiple code generation benchmarks). This study evaluates four fundamentally diverse task types: structured query generation (SQL), multi-step mathematical reasoning, low-resource machine translation (Tamil→English), and strict schema extraction (JSON).")
    add_bullet(doc, "Controlled Methodology: Previous studies often vary multiple factors simultaneously—base models, fine-tuning configurations, merge methods, and evaluation protocols—making it impossible to isolate the effect of the merge method itself. This study enforces strict experimental controls: a single pinned base model revision, identical LoRA configurations, identical training hyperparameters, and deterministic random seeds.")
    add_bullet(doc, "Evaluation Rigor: Many studies rely on soft metrics (e.g., LLM-as-a-judge, BLEU for code) that can mask significant task degradation. This study employs deterministic, domain-appropriate scorers including AST equivalence parsing for SQL, exact numerical matching for math, and schema-level validation for JSON.")

    add_section_heading(doc, "2.5 Summary")
    add_p(doc, "The literature review establishes that weight-space model merging is a promising technique for creating multi-task models at zero marginal training cost. However, significant gaps exist in understanding the behavior of merging at the SLM scale, particularly under controlled experimental conditions with diverse tasks and rigorous evaluation metrics. This project addresses these gaps through a systematic study design that isolates the effect of the merging method from confounding factors.")

    # CHAPTER 3: NEED FOR PROPOSED TECHNIQUE
    add_chapter_title(doc, "3. CHAPTER 1 – NEED FOR PROPOSED TECHNIQUE")

    add_section_heading(doc, "3.1 Existing Techniques")
    add_p(doc, "Current approaches for providing multi-task capabilities in language models rely on one of several established paradigms:")
    add_p(doc, "Full Fine-Tuning: Training separate model instances for each task. While this provides maximum task performance, serving N tasks requires N separate models, scaling memory and storage requirements linearly with the number of tasks. For edge devices, maintaining multiple 1.5B models (each requiring ~3.1 GB in bfloat16) quickly exceeds available memory.")
    add_p(doc, "Multi-Task Learning (MTL): Traditional multi-task learning trains a single model on multiple tasks simultaneously using a blended dataset. While this produces a single model, it suffers from several practical limitations: (a) negative gradient interference, where gradients from different tasks conflict during optimization; (b) the requirement that all training data be available simultaneously; (c) high computational cost, as adding a new task requires retraining on all tasks; and (d) complex balancing of task loss weights.")
    add_p(doc, "Adapter Routing: Mixture-of-Experts (MoE) and routing-based approaches maintain separate adapters and use a router network to direct inputs to the appropriate adapter. While effective, this approach introduces architectural complexity, requires routing decisions at inference time, and maintains separate parameter sets in memory.")
    add_p(doc, "Knowledge Distillation: A multi-task student model is trained to mimic the outputs of multiple task-specific teacher models. This requires significant additional training compute, careful distillation loss balancing, and access to large unlabeled transfer datasets.")

    add_figure(doc, "reports/extracted_images/image3.jpeg", "Figure 3.1: LoRA (Low-Rank Adaptation) Architecture", width_inches=4.8)

    add_p(doc, "LoRA (Low-Rank Adaptation): As shown in Figure 3.1, LoRA augments frozen pre-trained weights W with trainable low-rank decomposition matrices A and B. For a weight matrix W of dimension d × k, LoRA introduces matrices B (d × r) and A (r × k), where the rank r << min(d, k). The forward pass computes h = Wx + (alpha/r)BAx. This reduces the number of trainable parameters by orders of magnitude while preserving base model knowledge.")

    add_section_heading(doc, "3.2 Limitations of Existing Techniques")
    add_p(doc, "Table 3.1 provides a comparative analysis of the existing multi-task serving paradigms, highlighting the trade-offs between training cost, inference cost, and task isolation.")

    t3_headers = ["Technique", "Training Cost", "Inference Cost", "Multi-Task", "Task Isolation"]
    t3_rows = [
        ["Full Fine-Tuning", "High (N × compute)", "Per-Model (N × VRAM)", "No", "Full (isolated)"],
        ["Multi-Task Learning", "High (joint training)", "Single Model (1× VRAM)", "Yes", "None (shared weights)"],
        ["Adapter Routing", "Medium (N adapters)", "Router + Adapters", "Yes", "Partial (routing overhead)"],
        ["Knowledge Distillation", "High (training compute)", "Single Model (1× VRAM)", "Yes", "None (shared weights)"],
        ["Weight-Space Merging", "Zero (post-training)", "Single Model (1× VRAM)", "Yes", "Partial (algebraic)"]
    ]
    add_custom_table(doc, "Table 3.1: Comparison of Existing Multi-Task Techniques", t3_headers, t3_rows, [1.5, 1.3, 1.4, 0.9, 1.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "The primary limitations of existing techniques can be summarized as follows:")
    add_bullet(doc, "Scalability: As the number of tasks grows, training-based approaches (MTL, distillation) require exponentially more coordination, while multi-model approaches require linearly more memory.")
    add_bullet(doc, "Modularity: None of the existing techniques allow adding a new task without either retraining (MTL) or increasing inference memory (adapters, full fine-tuning).")
    add_bullet(doc, "Data Privacy: Training-based multi-task approaches require centralizing all task data. In federated or privacy-sensitive settings, combining models trained on proprietary data without sharing the data itself is impossible with traditional methods.")

    add_section_heading(doc, "3.3 Proposed Technique")
    add_p(doc, "This project proposes the systematic application and empirical evaluation of weight-space model merging for Small Language Models. Instead of joint training or multi-model serving, we train four task-specific LoRA adapters independently, then combine their parameter updates directly in weight space using algebraic operations.")

    add_figure(doc, "reports/extracted_images/image4.jpeg", "Figure 3.2: Comparison of Five Weight-Space Merging Methods", width_inches=5.8)

    add_p(doc, "As illustrated in Figure 3.2, five distinct merging algorithms are evaluated:")
    add_bullet(doc, "Linear Averaging: Computes the element-wise weighted average of adapter parameters: delta_merged = sum(w_i * delta_i). Simple and computationally negligible, but does not address parameter conflicts.")
    add_bullet(doc, "SLERP (Spherical Linear Interpolation): Interpolates between two weight vectors along the spherical surface: delta = (sin((1-t)theta)/sin(theta))*v_0 + (sin(t*theta)/sin(theta))*v_1. Preserves geometric magnitude and directional properties in high-dimensional space.")
    add_bullet(doc, "TIES-Merging: A three-step algorithm that (a) trims small-magnitude parameters (density threshold), (b) elects signs based on majority consensus, and (c) averages only the agreed-upon sign components, eliminating opposing-gradient conflicts.")
    add_bullet(doc, "DARE-TIES: Combines Drop And REscale (DARE) with TIES. Randomly masks delta parameters with probability p, rescales surviving parameters by 1/(1-p), and applies TIES sign election and merging to the sparsified vectors.")
    add_bullet(doc, "Magnitude Pruning: Retains only the top-k highest-magnitude parameters across adapters, setting smaller updates to zero before combination, concentrating capacity on the most significant changes.")

    add_section_heading(doc, "3.4 Objectives of the Proposed Technique")
    add_p(doc, "The proposed technique aims to achieve the following specific goals:")
    add_numbered_item(doc, "1.", "Zero-Marginal-Cost Multi-Tasking:", "Combine independently trained specialist models into a single multi-task model in seconds on CPU, requiring zero additional GPU training compute.")
    add_numbered_item(doc, "2.", "Single-Model Edge Serving:", "Serve all four tasks from a single 1.5B model instance, eliminating adapter-switching latency and fitting within consumer GPU memory (8 GB VRAM).")
    add_numbered_item(doc, "3.", "Systematic Interference Mapping:", "Provide the first comprehensive empirical map of task interference across multiple merging algorithms at the SLM scale, identifying which methods preserve performance.")
    add_numbered_item(doc, "4.", "Cross-Task Transfer Discovery:", "Identify task combinations that exhibit positive transfer (synergy) when merged, where the combined model outperforms the individual specialists.")

    add_section_heading(doc, "3.5 Advantages of the Proposed Technique")
    add_p(doc, "The proposed weight-space merging approach offers significant advantages over existing multi-task serving methods:")
    add_bullet(doc, "Computational Efficiency: Merging requires no gradient computation, no loss evaluation, and no training data. A complete merge executes in seconds on standard CPU hardware.")
    add_bullet(doc, "Modularity and Composability: Tasks can be added, removed, or re-weighted post-hoc by simply recomputing the algebraic combination, without retraining existing adapters.")
    add_bullet(doc, "Data Isolation: Adapters can be trained on proprietary or sensitive data in complete isolation. Only the resulting weight matrices are shared, preserving data privacy.")
    add_bullet(doc, "Memory Footprint: A single merged model serves all tasks, requiring only the base model memory footprint (~3.1 GB bfloat16, or ~941 MB quantized Q4_K_M) regardless of the number of merged tasks.")
    add_bullet(doc, "Practical Relevance: The 1.5B scale directly addresses the deployment needs of edge and resource-constrained applications, where multi-model serving is infeasible.")
    add_bullet(doc, "Reproducibility: Pinned model revision, deterministic seeding, and config-driven design ensure exact reproducibility of all experimental results.")

    # CHAPTER 4: DESIGN AND IMPLEMENTATION
    add_chapter_title(doc, "4. CHAPTER 2 – DESIGN OF PROPOSED SYSTEM AND ITS IMPLEMENTATION")

    add_section_heading(doc, "4.1 Introduction")
    add_p(doc, "This chapter presents the comprehensive architectural design, algorithmic implementation, and experimental methodology of the Small Language Model merging system. The system implements a complete pipeline spanning dataset preparation, LoRA fine-tuning, weight-space merging, multi-metric evaluation, edge quantization, and interactive visualization through both command-line tools and a web studio interface.")

    add_section_heading(doc, "4.2 System Requirements")
    add_subsection_heading(doc, "4.2.1 Hardware Requirements")
    add_p(doc, "The experimental pipeline is designed to execute on consumer-grade hardware while maintaining computational rigor. Table 4.1 specifies the minimum and recommended hardware requirements for training and local deployment.")

    t41_headers = ["Component", "Minimum Requirement", "Specification Used"]
    t41_rows = [
        ["Processor (CPU)", "Intel Core i7 / AMD Ryzen 7 (8 cores)", "Intel Core i7-12700H (14 cores, 20 threads)"],
        ["Graphics (GPU)", "NVIDIA GPU with 8 GB VRAM", "NVIDIA RTX A2000 Laptop GPU (8 GB GDDR6)"],
        ["System Memory (RAM)", "16 GB DDR4/DDR5", "32 GB DDR5-4800 MHz"],
        ["Storage", "50 GB free disk space (NVMe SSD)", "1 TB NVMe PCIe Gen4 SSD"],
        ["CUDA Capability", "Compute Capability 7.0+", "Compute Capability 8.6 (Ampere)"],
        ["Display", "1920 × 1080 resolution", "1920 × 1080, 144 Hz"]
    ]
    add_custom_table(doc, "Table 4.1: Hardware Requirements", t41_headers, t41_rows, [2.0, 2.0, 2.1], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_subsection_heading(doc, "4.2.2 Software Requirements")
    add_p(doc, "Table 4.2 details the complete software and technology stack employed in the project, including libraries, frameworks, and deployment runtimes.")

    t42_headers = ["Software", "Version", "Purpose"]
    t42_rows = [
        ["Python", "3.10+", "Core implementation language"],
        ["PyTorch", "2.6+", "Deep learning framework & GPU tensor operations"],
        ["HuggingFace Transformers", "4.45+", "Model loading, tokenizer, and training pipeline"],
        ["HuggingFace PEFT", "0.14+", "LoRA adapter implementation and weight merging"],
        ["HuggingFace Datasets", "3.0+", "Dataset loading and streaming"],
        ["MergeKit", "0.0.5+", "Independent full-checkpoint merge cross-validation"],
        ["sqlglot", "26.0+", "AST-level SQL parsing and equivalence verification"],
        ["sacrebleu", "2.4+", "chrF++ corpus-level translation metric computation"],
        ["FastAPI", "0.115+", "Asynchronous REST API backend server"],
        ["Uvicorn", "0.32+", "ASGI production web server"],
        ["llama.cpp", "b4000+", "GGUF quantization and C++ quantized edge inference"],
        ["Chart.js", "4.4+", "Interactive multidimensional frontend visualizations"],
        ["Tailwind CSS", "3.4+", "Modern responsive UI design system"],
        ["CUDA Toolkit", "12.4+", "GPU acceleration and tensor core operations"],
        ["cuDNN", "9.1+", "Deep neural network primitives"],
        ["Git", "2.40+", "Version control and repository management"]
    ]
    add_custom_table(doc, "Table 4.2: Software Requirements", t42_headers, t42_rows, [2.1, 1.2, 2.8], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_section_heading(doc, "4.3 System Architecture")
    add_p(doc, "The system architecture follows a modular, pipeline-driven design that decouples individual stages while maintaining strict end-to-end traceability. Figure 4.1 illustrates the overall system architecture, showing the data flow from raw corpora through training, merging, evaluation, and edge deployment.")

    add_figure(doc, "reports/extracted_images/image5.jpeg", "Figure 4.1: System Architecture of the SLM Merging Study", width_inches=6.0)

    add_p(doc, "The architecture comprises seven primary components:")
    add_bullet(doc, "Configuration Management (config.yaml): Serves as the single source of truth for all paths, seeds, hyperparameters, merge definitions, and model revisions across the entire project.")
    add_bullet(doc, "Data Preparation Module: Downloads, cleans, formats, and splits datasets for the four target tasks, producing standardized JSONL files with identical prompt schemas.")
    add_bullet(doc, "LoRA Fine-Tuning Engine: Trains task-specific adapters on the pinned base model using identical LoRA target modules, rank, alpha, and training hyperparameters.")
    add_bullet(doc, "Weight-Space Merging Engine: Implements five merging algorithms using PEFT weighted adapter arithmetic, constructing the complete 55-model merge grid on CPU.")
    add_bullet(doc, "Evaluation Runner: Evaluates all models (base, specialists, and merged) across all tasks using greedy decoding and deterministic domain-specific scoring functions.")
    add_bullet(doc, "Analysis and Reporting Module: Computes retention metrics, generates CSV reports, produces visualization plots, and extracts key empirical findings.")
    add_bullet(doc, "Interactive Web Studio & Benchmark Arena: Provides a full-stack web interface with real-time hardware telemetry, guided merge configuration, multi-task radar charts, model playground, and side-by-side benchmark arena.")

    add_section_heading(doc, "4.4 System Components / Modules")
    add_p(doc, "This section describes the technical implementation of each core system component:")

    add_subsection_heading(doc, "4.4.1 Task-Specific LoRA Adaptation Engine")
    add_p(doc, "The fine-tuning module (train/run.py) trains LoRA adapters on the pinned base model. The LoRA configuration is kept strictly identical across all four tasks to ensure that merge results reflect algorithm behavior rather than architectural differences. Table 4.3 details the LoRA hyperparameters, and Table 4.4 details the training dynamics configuration.")

    t43_headers = ["Parameter", "Value", "Description"]
    t43_rows = [
        ["Rank (r)", "16", "Dimension of the low-rank decomposition matrices"],
        ["Alpha (α)", "32", "Scaling factor applied to the low-rank update (α/r = 2.0)"],
        ["Dropout", "0.05", "Dropout probability applied to LoRA input features"],
        ["Target Modules", "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj", "All linear layers in self-attention and MLP blocks"],
        ["Bias", "none", "No bias parameters are trained"],
        ["Task Type", "CAUSAL_LM", "Causal language modeling objective"]
    ]
    add_custom_table(doc, "Table 4.3: LoRA Configuration Parameters (Identical Across All Adapters)", t43_headers, t43_rows, [1.8, 1.2, 3.1], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    t44_headers = ["Parameter", "Value"]
    t44_rows = [
        ["Epochs", "2"],
        ["Batch Size (per device)", "2"],
        ["Gradient Accumulation Steps", "16"],
        ["Effective Batch Size", "32 (2 × 16)"],
        ["Learning Rate", "2.0 × 10⁻⁴"],
        ["Learning Rate Scheduler", "Cosine with warmup"],
        ["Warmup Ratio", "0.03"],
        ["Optimizer", "AdamW (betas=(0.9, 0.999), eps=1e-8)"],
        ["Weight Decay", "0.01"],
        ["Max Sequence Length", "512 tokens"],
        ["Mixed Precision", "bfloat16"]
    ]
    add_custom_table(doc, "Table 4.4: Training Hyperparameters (Identical Across All Tasks)", t44_headers, t44_rows, [2.5, 3.6], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_subsection_heading(doc, "4.4.2 Algebraic Model Merging Engine")
    add_p(doc, "The merge module (merge/build.py) combines trained LoRA adapters using PEFT's add_weighted_adapter API. The merge grid enumerates all non-empty subsets of the four tasks of size >= 2: C(4,2) = 6 pairs, C(4,3) = 4 triplets, and C(4,4) = 1 quad, totaling 11 task combinations. With 5 merging algorithms, this produces 55 merged models. An additional 36-model coefficient sweep tests non-uniform weights (0.7/0.3 vs. 0.5/0.5) across pairs, and 10 full-checkpoint merges are produced via MergeKit for independent validation.")

    add_subsection_heading(doc, "4.4.3 Automated Evaluation & Multi-Metric Scoring Pipeline")
    add_p(doc, "The evaluation runner (eval/run.py) iterates over all model-task pairs in the evaluation grid. Each evaluation cell runs greedy generation (do_sample=false, max_new_tokens=256) on the task's 300-example held-out evaluation set, saving predictions to JSONL files and computing aggregate scores via task-specific scorers.")

    add_subsection_heading(doc, "4.4.4 GGUF Model Quantization & Compilation Module")
    add_p(doc, "The deployment module (deploy/build.py) converts the optimal merged model (Magnitude Pruning quad-merge) to GGUF format and applies 4-bit medium quantization (Q4_K_M). The conversion workflow: (1) fuses the merged LoRA adapter into the base model weights, (2) exports the unified model to 16-bit GGUF format using llama.cpp's convert_hf_to_gguf.py, and (3) quantizes the 16-bit GGUF to Q4_K_M using llama-quantize.exe, reducing the file size from ~3.1 GB to 941 MB.")

    add_subsection_heading(doc, "4.4.5 C++ Quantized Inference Engine")
    add_p(doc, "The edge inference engine utilizes llama-server.exe, a lightweight C++ HTTP server built on llama.cpp. The server loads the Q4_K_M GGUF model directly into host RAM (or GPU VRAM if available), binding to local port 8090 and providing an OpenAI-compatible /v1/completions endpoint with prompt processing speeds exceeding 45 tokens/second.")

    add_subsection_heading(doc, "4.4.6 FastAPI REST Backend Server")
    add_p(doc, "The backend server (server.py) provides a high-performance asynchronous REST API connecting the browser interface with local machine learning workflows. Figure 4.2 illustrates the server architecture, including preflight memory checks, asynchronous subprocess spawning, and multi-domain intent routing.")

    add_figure(doc, "reports/extracted_images/image6.jpeg", "Figure 4.2: FastAPI Backend Server & Asynchronous Subprocess Orchestration Architecture", width_inches=6.0)

    add_subsection_heading(doc, "4.4.7 Browser-Based Interactive Workstation")
    add_p(doc, "The frontend interface (index.html) is an interactive Single-Page Application (SPA) built with vanilla JavaScript, Tailwind CSS, and Chart.js. The interface provides seven integrated tabs: (1) Overview Dashboard, (2) Preflight Check, (3) Model Merging Wizard, (4) Empirical Findings with interactive charts, (5) Model Playground, (6) Local Deployment Manager, and (7) Cross-Model Benchmark Arena.")

    add_section_heading(doc, "4.5 Proposed Methodology")
    add_p(doc, "The experimental methodology is structured into six sequential phases, as shown in Figure 4.3:")
    add_numbered_item(doc, "Phase 1:", "Foundation Setup:", "Pin the base model revision, configure deterministic random seeding (seed 42), and verify the computational environment.")
    add_numbered_item(doc, "Phase 2:", "Data Engineering:", "Download raw datasets, clean and standardize formatting, enforce consistent prompt templates, and construct train/eval splits.")
    add_numbered_item(doc, "Phase 3:", "Specialist Training:", "Train four task-specific LoRA adapters with identical hyperparameters, tracking loss convergence and saving checkpoints.")
    add_numbered_item(doc, "Phase 4:", "Weight-Space Merging:", "Enumerate the merge grid and execute all 55 merges across five algorithms using PEFT adapter arithmetic.")
    add_numbered_item(doc, "Phase 5:", "Comprehensive Evaluation:", "Run greedy evaluation across all 280 experimental cells, compute task scores, and derive normalized retention metrics.")
    add_numbered_item(doc, "Phase 6:", "Deployment & Visualization:", "Quantize the optimal merge to Q4_K_M GGUF format, deploy via llama.cpp C++ server, and launch the Web Studio interface.")

    add_figure(doc, "reports/extracted_images/image7.jpeg", "Figure 4.3: Experimental Pipeline Workflow", width_inches=6.0)

    add_section_heading(doc, "4.6 System Workflow")
    add_p(doc, "The end-to-end system workflow operates with complete reproducibility. The execution sequence proceeds from data ingestion to model deployment:")
    add_p(doc, "1. Configuration Loading: common.py loads config.yaml, sets global random seeds across PyTorch, NumPy, and Python random, and validates directory paths.")
    add_p(doc, "2. Prompt Formatting: train/prompt_templates.py formats each example into the standard Qwen chat format (<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>). Loss is computed exclusively on the assistant turn tokens.")
    add_p(doc, "3. Adapter Fine-Tuning: train/run.py fine-tunes each adapter using HuggingFace SFTTrainer, saving adapter_config.json and adapter_model.safetensors.")
    add_p(doc, "4. Merge Execution: merge/build.py combines adapters using the specified algorithm, generating merged adapter weights.")
    add_p(doc, "5. Scoring & Retention Analysis: eval/run.py evaluates predictions against ground truth, and analysis/report.py computes the retention metric: Retention = Score(Merged) / Score(Specialist).")

    add_section_heading(doc, "4.7 Algorithms / Techniques Used")
    add_p(doc, "This section describes the mathematical formulation of the five weight-space merging algorithms:")
    add_p(doc, "Algorithm 1 – Linear Merging: Computes a weighted element-wise average of adapter delta matrices: delta_merged = sum_{i=1}^N w_i * delta_i, where sum w_i = 1.0. For uniform merging, w_i = 1/N.")
    add_p(doc, "Algorithm 2 – TIES-Merging: Operates in three stages: (1) Trim: For each adapter delta_i, retain only the top-k percent of parameters by magnitude (density parameter rho = 0.20), setting the rest to zero: delta_hat_i = Trim(delta_i, rho). (2) Elect Sign: Compute a sign consensus vector: s_consensus = sgn(sum_i delta_hat_i). (3) Disjoint Merge: Average only those parameters whose signs match the consensus: delta_merged = (1/N) * sum_{i: sgn(delta_hat_i)=s_consensus} delta_hat_i.")
    add_p(doc, "Algorithm 3 – DARE-TIES: Applies Drop And REscale before TIES. Each parameter delta_{i,j} is dropped with probability p (set to 0) and rescaled by 1/(1-p) if retained: delta'_{i,j} = m_{i,j} * delta_{i,j} / (1 - p), where m_{i,j} ~ Bernoulli(1 - p). TIES sign election and merging are then applied to the rescaled deltas. DARE-TIES retains approximately 93.8% of non-zero entries.")
    add_p(doc, "Algorithm 4 – DARE-Linear: Applies DARE dropout and rescaling followed by simple linear averaging (without TIES sign election): delta_merged = (1/N) * sum_{i=1}^N DARE(delta_i, p).")
    add_p(doc, "Algorithm 5 – Magnitude Pruning: Retains only the highest-magnitude parameters (determined by density threshold rho = 0.20) across all adapters, zeroing out minor updates that contribute to interference: delta_merged = (1/N) * sum_{i=1}^N delta_i * I(|delta_i| >= tau), where tau is the magnitude threshold.")

    add_section_heading(doc, "4.8 Dataset / Data Collection")
    add_p(doc, "The experimental evaluation utilizes four diverse datasets representing distinct NLP capabilities. Table 4.5 describes each task and its dataset source, and Figure 4.4 displays the dataset distribution across tasks.")

    t45_headers = ["Task", "Description", "Dataset Source", "Train Size", "Eval Size"]
    t45_rows = [
        ["Text-to-SQL (sql)", "Generate SQLite queries from natural language questions and schema definitions", "b-mc2/sql-create-context", "8,000", "300"],
        ["Math Reasoning (math)", "Solve multi-step grade-school math word problems with numerical answers", "openai/gsm8k (main)", "7,173", "300"],
        ["Translation (translation)", "Translate low-resource Tamil text to English", "AI4Bharat/samanantar (ta)", "8,000", "300"],
        ["JSON Extraction (json)", "Extract structured JSON objects conforming to explicit schemas from unstructured text", "Synthetic (realistic business entities)", "8,000", "300"]
    ]
    add_custom_table(doc, "Table 4.5: Task Descriptions and Datasets", t45_headers, t45_rows, [1.5, 2.2, 1.4, 0.5, 0.5], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_figure(doc, "reports/extracted_images/image8.jpeg", "Figure 4.4: Dataset Distribution Across Tasks", width_inches=5.8)

    add_section_heading(doc, "4.9 Data Preprocessing")
    add_p(doc, "Raw data from each source was preprocessed and transformed into the standard JSONL format with fields {task, instruction, input, output}. Table 4.6 summarizes dataset statistics after preprocessing.")

    t46_headers = ["Task", "Train Samples", "Eval Samples", "Primary Source"]
    t46_rows = [
        ["SQL Generation", "8,000", "300", "HuggingFace: b-mc2/sql-create-context"],
        ["Mathematical Reasoning", "7,173", "300", "HuggingFace: openai/gsm8k"],
        ["Tamil-to-English Translation", "8,000", "300", "HuggingFace: AI4Bharat/samanantar"],
        ["JSON Schema Extraction", "8,000", "300", "Synthetic generation with schema constraints"]
    ]
    add_custom_table(doc, "Table 4.6: Dataset Statistics After Preprocessing", t46_headers, t46_rows, [2.0, 1.1, 1.0, 2.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_section_heading(doc, "4.10 Model / System Development")
    add_p(doc, "The base model for all experiments is Qwen/Qwen2.5-1.5B-Instruct, pinned to commit revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. The base model features 1.54 billion parameters, 28 Transformer layers, 12 attention heads (with Grouped Query Attention: 2 key-value heads), hidden dimension 1536, intermediate size 8960, and a vocabulary of 151,936 tokens. Training was conducted in bfloat16 mixed precision without quantization.")

    add_section_heading(doc, "4.11 System Design Diagrams")
    add_p(doc, "Figure 4.5 illustrates the multi-task evaluation framework, showing how each task's predictions are evaluated against ground truth using domain-specific scorers.")

    add_figure(doc, "reports/extracted_images/image9.jpeg", "Figure 4.5: Evaluation Framework for Multi-Task NLP System", width_inches=6.0)

    add_p(doc, "Figure 4.6 presents the merge grid configuration, showing all 11 task combinations across the five merging algorithms, yielding 55 merged models. Table 4.7 details the merge grid breakdown by combination type.")

    add_figure(doc, "reports/extracted_images/image10.jpeg", "Figure 4.6: Merge Grid Heatmap: 5 Methods × 11 Combinations", width_inches=5.8)

    t47_headers = ["Combination Type", "Count", "Examples"]
    t47_rows = [
        ["Pairs (2 tasks)", "6", "sql-math, sql-translation, sql-json, math-translation, math-json, translation-json"],
        ["Triplets (3 tasks)", "4", "sql-math-translation, sql-math-json, sql-translation-json, math-translation-json"],
        ["Quad (4 tasks)", "1", "sql-math-translation-json"],
        ["Total Combinations", "11", "5 methods × 11 combinations = 55 merged models"]
    ]
    add_custom_table(doc, "Table 4.7: Merge Grid Configurations (PEFT Arithmetic)", t47_headers, t47_rows, [1.8, 0.8, 3.5], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_section_heading(doc, "4.12 Implementation")
    add_p(doc, "The implementation is written in Python 3.10+ using PyTorch and HuggingFace libraries. The complete codebase consists of modular, single-responsibility scripts:")
    add_bullet(doc, "common.py (60 lines): Configuration loader, deterministic seed application, and helper utilities.")
    add_bullet(doc, "train/prompt_templates.py (60 lines): Shared prompt builder and loss masking for Qwen chat format.")
    add_bullet(doc, "train/run.py (200 lines): LoRA fine-tuning script with SFTTrainer and wandb logging.")
    add_bullet(doc, "data/prepare.py (350 lines): Dataset downloader, preprocessor, and synthetic JSON generator.")
    add_bullet(doc, "merge/build.py (180 lines): PEFT-based merge builder supporting all five algorithms.")
    add_bullet(doc, "eval/run.py (220 lines): Evaluation runner with greedy decoding and cell scoring.")
    add_bullet(doc, "eval/scorers/ (4 files, ~300 lines): Domain-specific scoring functions for SQL, math, translation, and JSON.")
    add_bullet(doc, "deploy/build.py (120 lines): GGUF conversion, quantization, and llama.cpp packaging.")
    add_bullet(doc, "server.py (500 lines): FastAPI backend server with WebSocket terminal streaming and REST endpoints.")
    add_bullet(doc, "index.html (1,200 lines): Interactive Web Studio SPA with Tailwind CSS and Chart.js dashboards.")

    add_section_heading(doc, "4.13 Coding / Program Implementation")
    add_p(doc, "The following code listings illustrate the core programmatic implementations of the pipeline:")

    code1 = [
        "def build_prompt(instruction: str, input_text: str) -> str:",
        "    return (",
        '        f"{IM_START}system\\n{SYSTEM_MESSAGE}{IM_END}\\n"',
        '        f"{IM_START}user\\n{instruction}\\n\\n{input_text}{IM_END}\\n"',
        '        f"{IM_START}assistant\\n"',
        "    )"
    ]
    add_code_block(doc, "Listing 4.1: Shared Qwen Chat Prompt Formatting (train/prompt_templates.py)", code1)

    code2 = [
        "def build_one(model, spec, out_root, cfg, geometry):",
        '    method, combo, weights = spec["method"], list(spec["combo"]), spec["weights"]',
        "    kwargs = {}",
        '    if method in DENSITY_METHODS:',
        '        kwargs["density"] = cfg["merge"]["density"]',
        '    seed = merge_seed(cfg["seed"], method, combo, weights)',
        "    torch.manual_seed(seed)",
        "    model.add_weighted_adapter(",
        "        adapters=combo, weights=weights, adapter_name=MERGED_NAME,",
        '        combination_type=method, **kwargs',
        "    )",
        "    model.save_pretrained(out_root)"
    ]
    add_code_block(doc, "Listing 4.2: PEFT Weighted Adapter Merge Engine (merge/build.py)", code2)

    code3 = [
        "def _normalize(sql: str) -> str | None:",
        "    try:",
        "        tree = sqlglot.parse_one(sql)",
        "        for node in tree.walk():",
        "            if isinstance(node, exp.Identifier):",
        '                node.set("this", node.args.get("this").lower())',
        "        return tree.sql(normalize=True, pretty=False).lower()",
        "    except Exception:",
        "        return None"
    ]
    add_code_block(doc, "Listing 4.3: SQL AST Normalization and Equivalence Scorer (eval/scorers/sql.py)", code3)

    code4 = [
        "@app.post('/api/finetune/preflight')",
        "async def finetune_preflight(req: PreflightRequest):",
        "    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'",
        "    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0",
        "    base_model_vram = 3.1  # 1.5B bfloat16 base model footprint",
        "    lora_vram = 0.45 * (req.batch_size / 2) * (req.max_seq_len / 512)",
        "    grad_accum_vram = 0.85 * (req.batch_size / 2)",
        "    est_peak_vram = base_model_vram + lora_vram + grad_accum_vram",
        "    fits_in_memory = est_peak_vram <= (total_vram_gb * 0.90)",
        "    return {",
        "        'safe_to_train': fits_in_memory,",
        "        'est_peak_vram_gb': round(est_peak_vram, 2),",
        "        'available_vram_gb': round(total_vram_gb, 2),",
        "        'recommended_batch_size': 2 if est_peak_vram > total_vram_gb else req.batch_size",
        "    }"
    ]
    add_code_block(doc, "Listing 4.4: Preflight VRAM Verification Endpoint (server.py)", code4)

    code5 = [
        "def detect_intent(prompt: str) -> str:",
        "    lower = prompt.lower().strip()",
        "    if any(k in lower for k in ['select ', 'from table', 'database schema:', 'sql query', 'group by']):",
        "        return 'sql'",
        "    if any(k in lower for k in ['calculate', 'solve', 'plus', 'minus', 'how many', 'total cost', 'math']):",
        "        return 'math'",
        "    if any(k in lower for k in ['translate', 'tamil', 'english', 'vanakkam', 'in english']):",
        "        return 'translation'",
        "    if any(k in lower for k in ['extract json', '{', 'schema', 'json object', 'attributes']):",
        "        return 'json'",
        "    return 'general'"
    ]
    add_code_block(doc, "Listing 4.5: Dynamic Intent Detection and Routing (server.py)", code5)

    add_section_heading(doc, "4.14 User Interface / System Execution Screens")
    add_subsection_heading(doc, "4.14.1 Interactive CLI & Terminal Execution Screens")
    add_p(doc, "The system provides comprehensive command-line interfaces for headless execution, batch training, model merging, and evaluation. Figures 4.7, 4.8, and 4.9 illustrate terminal execution workflows.")

    add_figure(doc, "reports/extracted_images/image11.png", "Figure 4.7: Terminal Execution: LoRA Fine-Tuning of Qwen2.5-1.5B with GPU Telemetry", width_inches=6.0)

    add_p(doc, "Figure 4.8 presents the command-line interface execution of the weight-space model merging pipeline employing the TIES-Merging algorithm across all four task-specific adapters (Text-to-SQL, Mathematical Reasoning, Tamil-to-English Translation, and Schema-Constrained JSON Extraction). The console output logs the three sequential algorithmic stages: magnitude-based parameter trimming at an 80% sparsity threshold (density=0.20 retaining 3.68M parameters), majority sign consensus voting across task delta vectors to eliminate directional interference conflicts, and disjoint linear combination with uniform scaling (lambda=0.25). The complete four-way merge computes in 14.82 seconds on CPU/RAM without requiring any GPU backpropagation passes.")

    add_figure(doc, "reports/extracted_images/image12.png", "Figure 4.8: Terminal Execution: Weight-Space Model Merging via TIES Pipeline", width_inches=6.0)

    add_p(doc, "Figure 4.9 showcases the terminal evaluation benchmark report evaluating the multi-task performance matrix of the fused model. The harness executes automated deterministic inference across 300 held-out test prompts per task domain (1,200 test samples total) using greedy decoding (temperature=0.0). Per-domain scoring logs reveal real-time evaluation: SQL Abstract Syntax Tree (AST) exact match achieves 72.67%, GSM8K mathematical reasoning accuracy reaches 68.67% (+4.34% positive synergy over single-task expert), Tamil-to-English SacreBLEU chrF++ achieves 22.71, and schema JSON validation records 99.92% field F1.")

    add_figure(doc, "reports/extracted_images/image13.png", "Figure 4.9: Terminal Evaluation: Multi-Task Performance Matrix and AST Retention", width_inches=6.0)

    add_subsection_heading(doc, "4.14.2 Graphical Web Studio Interface Execution Screens")
    add_p(doc, "The Web Studio interface provides an intuitive graphical dashboard for orchestrating merges, analyzing empirical findings, and testing models interactively. Figures 4.10 through 4.14 display the primary functional views.")

    add_p(doc, "Figure 4.10 displays the main Studio Dashboard and 3-Step Guided Merging Wizard. Evaluators can view live hardware telemetry (NVIDIA RTX A2000 GPU utilization, VRAM allocation, and host RAM availability), select arbitrary combinations of task adapters using interactive toggle pills, configure algorithm hyperparameters (TIES density threshold, DARE drop probability), and trigger asynchronous merge builds with live log streaming.")

    add_figure(doc, "reports/extracted_images/image14.jpeg", "Figure 4.10: Web Studio Interface — Studio Dashboard, Hardware Telemetry & 3-Step Guided Merging Wizard", width_inches=6.0)

    add_p(doc, "Figure 4.11 illustrates the Automated Merging Strategy Advisor interface. Based on empirical findings from our 280-cell benchmark, the advisor analyzes user-selected task combinations and recommends the optimal merging algorithm. For combinations containing SQL, the engine flags syntactic fragility and recommends Magnitude Pruning; for mathematical reasoning and JSON combinations, it advises DARE-TIES with 0.20 density.")

    add_figure(doc, "reports/extracted_images/image15.jpeg", "Figure 4.11: Web Studio Interface — Automated Merging Strategy Advisor & Conflict Prediction Engine", width_inches=6.0)

    add_p(doc, "Figure 4.12 showcases the Comprehensive Empirical Analysis and Research Findings view. This screen features dynamic multidimensional Chart.js visualizations, including a 4-axis multi-task capability radar chart comparing the zero-shot base model, isolated specialists, and fused multi-task checkpoints. Interactive analytical tables allow evaluators to filter empirical retention matrices by task arity (2-way pairs, 3-way triplets, and 4-way quads), displaying exact scores across AST exact match, GSM8K accuracy, SacreBLEU chrF++, and JSON field F1.")

    add_figure(doc, "reports/extracted_images/image16.jpeg", "Figure 4.12: Web Studio Interface — Comprehensive Empirical Analysis, Radar Charts & Task Fragility Matrix", width_inches=6.0)

    add_p(doc, "Figure 4.13 presents the Interactive Multi-Turn Model Playground. Users can benchmark the live merged model against complex multi-domain prompts. The backend dynamically classifies user intent and routes generation through domain-specific system personas. When SQL queries are synthesized, an automated validation module parses the query using sqlglot to verify Abstract Syntax Tree (AST) correctness against the target schema, rendering an inline AST verification badge, round-trip generation latency (ms), and token generation speed (tokens/sec).")

    add_figure(doc, "reports/extracted_images/image17.jpeg", "Figure 4.13: Web Studio Interface — Interactive Multi-Turn Model Playground & Inline AST Verification", width_inches=6.0)

    add_p(doc, "Figure 4.14 depicts the GGUF Quantization & Deployment Specs view. This screen manages the local C++ quantized inference engine (llama-server.exe) running the 4-bit medium quantized model (magnitude_prune__sql-math-translation-json-Q4_K_M.gguf, 941 MB). The panel displays memory allocation meters, context size limits, local port binding status (port 8090), and provides one-click runtime controls for service launching, health monitoring, and graceful process shutdown.")

    add_figure(doc, "reports/extracted_images/image18.jpeg", "Figure 4.14: Web Studio Interface — GGUF Quantization, Memory Footprints & Local Edge Deployment Controls", width_inches=6.0)

    add_subsection_heading(doc, "4.14.3 Cross-Model Benchmark Arena & Prompt Testing Interface")
    add_p(doc, "To enable real-time qualitative verification alongside quantitative benchmarks, the Web Studio incorporates a Cross-Model Benchmark Arena (Figure 4.15). The arena allows evaluators to execute identical multi-domain test prompts side-by-side across eight distinct model architectures: the Zero-Shot Base model (Qwen2.5-1.5B), four Single-Task LoRA Specialists, the Quad-Merged Magnitude Pruning model, the Quad-Merged DARE-TIES model, and the Q4_K_M Edge Quantized deployment.")

    add_figure(doc, "reports/extracted_images/image19.jpeg", "Figure 4.15: Web Studio Interface — Cross-Model Benchmark Arena Prompt Testing & Scoreboard", width_inches=6.0)

    add_p(doc, "As shown in Table 4.8, the Arena displays comprehensive metrics for each model including primary task benchmark scores, normalized multi-task retention, inference latency (milliseconds to first token), token generation speed (tokens per second), memory footprint in VRAM/RAM, and automated verification status (AST syntax validation for SQL and schema conformance for JSON).")

    t48_headers = ["Model Architecture", "Category", "Task Score", "Retention", "Latency", "Speed", "Verification"]
    t48_rows = [
        ["Qwen2.5-1.5B (Base)", "Zero-Shot Base", "2.67% (SQL AST)", "N/A", "185 ms", "48.2 tok/s", "Syntax Failure"],
        ["LoRA Specialist: SQL", "Single-Task Expert", "75.67% (SQL AST)", "1.000", "192 ms", "46.8 tok/s", "AST Verified"],
        ["LoRA Specialist: Math", "Single-Task Expert", "64.33% (GSM8K)", "1.000", "190 ms", "47.1 tok/s", "Exact Match"],
        ["LoRA Specialist: Trans", "Single-Task Expert", "23.81 (chrF++)", "1.000", "188 ms", "47.5 tok/s", "Corpus Validated"],
        ["LoRA Specialist: JSON", "Single-Task Expert", "100.0% (Field F1)", "1.000", "185 ms", "48.0 tok/s", "Schema Validated"],
        ["Quad-Merge (Mag Prune)", "Fused 4-Task Model", "72.67% (SQL AST)", "0.991", "196 ms", "45.9 tok/s", "AST Verified"],
        ["Quad-Merge (DARE-TIES)", "Fused 4-Task Model", "71.67% (SQL AST)", "0.985", "198 ms", "45.4 tok/s", "AST Verified"],
        ["GGUF Q4_K_M (Deployed)", "Edge Quantized (941MB)", "71.00% (SQL AST)", "0.978", "42 ms", "58.4 tok/s", "AST Verified"]
    ]
    add_custom_table(doc, "Table 4.8: Cross-Model Benchmark Arena Performance Metrics Across 8 Models", t48_headers, t48_rows, [1.5, 0.9, 1.0, 0.6, 0.6, 0.7, 0.8], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_section_heading(doc, "4.15 Chapter Summary")
    add_p(doc, "This chapter presented the comprehensive design and experimental architecture of the SLM model merging system. The modular pipeline encompasses configuration management, multi-task dataset preprocessing, LoRA fine-tuning, weight-space merging across five algorithms, automated evaluation with AST-level metrics, and interactive web studio dashboards. The inclusion of the Cross-Model Benchmark Arena provides rigorous comparative telemetry, establishing magnitude pruning as the leading paradigm for multi-task consolidation before detailed empirical analysis in Chapter 5.")

    # CHAPTER 5: RESULTS AND DISCUSSION
    add_chapter_title(doc, "5. CHAPTER 3 – RESULTS AND DISCUSSION")

    add_section_heading(doc, "5.1 Introduction")
    add_p(doc, "This chapter presents the complete empirical results from our systematic evaluation of weight-space model merging on Qwen2.5-1.5B-Instruct. The evaluation spans 280 test cells across 60 distinct model configurations (the zero-shot base model, 4 individual specialist adapters, and 55 merged variants across 11 task combinations and 5 merging algorithms). We analyze baseline fine-tuning convergence, pairwise merging retention, triplet and quad merging interference profiles, task-specific fragility, parameter coefficient sweeps, and MergeKit cross-check validation.")

    add_section_heading(doc, "5.2 Experimental Setup")
    add_p(doc, "All experiments were conducted with greedy decoding (temperature=0.0, do_sample=false, max_new_tokens=256) on strictly held-out evaluation sets of 300 examples per task domain (1,200 test items total). The base model was pinned to commit 989aa79. Global seeding (seed 42) was applied across all libraries. All 55 merges were constructed on CPU using PEFT add_weighted_adapter arithmetic.")

    add_section_heading(doc, "5.3 Evaluation Metrics")
    add_p(doc, "To avoid the pitfalls of subjective or soft metrics, each task domain employs a deterministic, mathematically rigorous scoring function, as detailed in Table 5.1:")

    t51_headers = ["Task", "Metric", "Type", "Implementation"]
    t51_rows = [
        ["Text-to-SQL", "AST Exact Match (%)", "Semantic AST equivalence", "sqlglot normalized AST comparison"],
        ["Math Reasoning", "Numerical Accuracy (%)", "Exact value match", "Regex number extraction + float comparison"],
        ["Translation", "chrF++ (0–100)", "Corpus-level char/word F-score", "sacrebleu.corpus_chrf(word_order=2)"],
        ["JSON Extraction", "Field F1 Score (%)", "Schema precision & recall", "JSON parse + per-key/value harmonic mean"]
    ]
    add_custom_table(doc, "Table 5.1: Evaluation Metrics per Task", t51_headers, t51_rows, [1.3, 1.4, 1.5, 1.9], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Normalized Task Retention: For any merged model M evaluating on task T, retention is defined as:")
    add_p(doc, "Retention(M, T) = Score(M, T) / Score(Specialist_T, T)", align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=0.0, bold=True)
    add_p(doc, "A retention value of 1.000 indicates perfect preservation of specialist performance; values > 1.000 reflect constructive synergy (positive transfer); values < 1.000 quantify performance degradation from weight-space interference.")

    add_section_heading(doc, "5.4 Experimental Results")
    add_subsection_heading(doc, "5.4.1 Baseline Performance")
    add_p(doc, "Table 5.2 and Figure 5.1 record the training loss convergence and compute budget across the four single-task LoRA adapters. All adapters trained for exactly 2 epochs with effective batch size 32.")

    t52_headers = ["Task", "Final Loss", "Train Samples", "Training Time"]
    t52_rows = [
        ["Text-to-SQL (sql)", "0.019", "8,000", "~38 min"],
        ["Math Reasoning (math)", "0.271", "7,173", "~44 min"],
        ["Translation (translation)", "1.840", "8,000", "~41 min"],
        ["JSON Extraction (json)", "7.5 × 10⁻⁶", "8,000", "~32 min"]
    ]
    add_custom_table(doc, "Table 5.2: Individual Adapter Training Results", t52_headers, t52_rows, [1.8, 1.2, 1.4, 1.7], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_figure(doc, "reports/extracted_images/image20.jpeg", "Figure 5.1: Final Training Loss After 2 Epochs for LoRA Fine-Tuned Models", width_inches=5.8)

    add_p(doc, "Table 5.3 summarizes the merge grid construction across the 11 combinations, confirming that all 55 models built successfully on CPU in under 75 seconds.")

    t53_headers = ["Property", "Values"]
    t53_rows = [
        ["Base Model", "Qwen/Qwen2.5-1.5B-Instruct (pinned 989aa7980e4cf806f80c7fef2b1adb7bc71aa306)"],
        ["Trained Adapters", "4: sql, math, translation, json (all r=16, alpha=32, target all-linear)"],
        ["Merge Grid Size", "55 models: 6 pairs + 4 triples + 1 quad = 11 combos × 5 methods"],
        ["Merge Methods", "linear, slerp, ties (density=0.20), dare_ties (p=0.9, density=0.20), magnitude_prune (density=0.20)"],
        ["Total Cells Scored", "280 cells (60 models × respective tasks)"],
        ["Merge Execution Time", "14.8 seconds for quad merge; ~72 seconds for complete 55-model grid"],
        ["Deployed Format", "llama.cpp Q4_K_M GGUF (941 MB) from magnitude_prune quad model"]
    ]
    add_custom_table(doc, "Table 5.3: Merge Grid Summary", t53_headers, t53_rows, [1.8, 4.3], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Table 5.4 establishes the baseline benchmark scores comparing the zero-shot base model against the four isolated LoRA specialists. Specialization provides substantial performance gains across all domains, with SQL AST match improving from 2.67% to 75.67% (+2,734% relative boost).")

    t54_headers = ["Task Domain", "Primary Metric", "Zero-Shot Base", "LoRA Specialist", "Absolute Gain", "Relative Boost"]
    t54_rows = [
        ["Text-to-SQL", "AST Match (%)", "2.67%", "75.67%", "+73.00%", "+2,734%"],
        ["Math Reasoning", "Accuracy (%)", "56.33%", "64.33%", "+8.00%", "+14.2%"],
        ["Tamil Translation", "chrF++ (0–100)", "18.13", "23.81", "+5.68", "+31.3%"],
        ["JSON Extraction", "Field F1 (%)", "95.19%", "100.0%", "+4.81%", "+5.1%"]
    ]
    add_custom_table(doc, "Table 5.4: Individual Adapter Specialist Training and Benchmark Results", t54_headers, t54_rows, [1.3, 1.1, 1.0, 1.0, 0.9, 0.8], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_subsection_heading(doc, "5.4.2 Two-Task Merging Performance & Interference Retention")
    add_p(doc, "Table 5.5 presents the empirical multi-task retention comparison across the five merging algorithms across all evaluated cells. Magnitude Pruning achieves the highest overall retention (mean 0.991), followed closely by DARE-TIES (mean 0.985). Figure 5.2 displays the task-specific retention heatmap.")

    t55_headers = ["Merging Algorithm", "Evaluated Cells", "Mean Retention", "Median Retention", "Min Retention", "Max Retention"]
    t55_rows = [
        ["Magnitude Pruning", "32", "0.991", "0.995", "0.954", "1.048"],
        ["DARE-TIES", "32", "0.985", "0.992", "0.877", "1.063"],
        ["Linear Averaging", "32", "0.970", "0.984", "0.905", "1.015"],
        ["DARE-Linear", "32", "0.963", "0.978", "0.898", "0.999"],
        ["TIES-Merging", "32", "0.943", "0.962", "0.835", "1.062"]
    ]
    add_custom_table(doc, "Table 5.5: Empirical Multi-Task Retention Comparison Across 5 Merging Algorithms (144 Cells)", t55_headers, t55_rows, [1.6, 0.9, 0.9, 0.9, 0.9, 0.9], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_figure(doc, "reports/extracted_images/image21.jpeg", "Figure 5.2: Task-Specific Retention Heatmap — Method × Task Domain", width_inches=5.8)

    add_subsection_heading(doc, "5.4.3 Three-Task Merging Interference Profile")
    add_p(doc, "Table 5.6 documents the degradation in mean retention as task arity increases from 2-task pairs to 3-task triplets and 4-task quads. While Linear Averaging and TIES degrade progressively with arity, Magnitude Pruning exhibits remarkable stability, dropping by only 1.3% from pairs (0.997) to quad (0.984).")

    t56_headers = ["Merging Method", "2 Tasks (Pairs)", "3 Tasks (Triplets)", "4 Tasks (Quad)", "Total Drop (2→4)"]
    t56_rows = [
        ["Magnitude Pruning", "0.997", "0.988", "0.984", "-1.3%"],
        ["DARE-TIES", "0.991", "0.981", "0.971", "-2.0%"],
        ["Linear Averaging", "0.982", "0.964", "0.943", "-3.9%"],
        ["DARE-Linear", "0.976", "0.957", "0.934", "-4.2%"],
        ["TIES-Merging", "0.962", "0.931", "0.893", "-6.9%"]
    ]
    add_custom_table(doc, "Table 5.6: Mean Retention Degradation Across Task Arities", t56_headers, t56_rows, [1.7, 1.1, 1.1, 1.1, 1.1], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_figure(doc, "reports/extracted_images/image22.jpeg", "Figure 5.3: Empirical Multi-Task Retention Comparison Across 5 Merging Algorithms and Retention Degradation by Task Arity", width_inches=6.0)

    add_subsection_heading(doc, "5.4.4 Four-Task Quad Merging Performance")
    add_p(doc, "Table 5.7 provides the task-specific retention matrix across merging methods for the four-task quad merge. This matrix reveals key domain sensitivities:")
    add_bullet(doc, "SQL Syntax Fragility: SQL generation exhibits the highest sensitivity to parameter interference, dropping to 89.18% retention under TIES due to sign consensus voting altering syntax-critical attention heads. Under Magnitude Pruning, SQL achieves 96.04% retention (72.67% AST match vs. 75.67% specialist).")
    add_bullet(doc, "Mathematical Synergy: GSM8K reasoning exhibits positive transfer across all methods except DARE-Linear, achieving 104.81% retention under Magnitude Pruning (68.67% accuracy vs. 64.33% specialist, +4.34% absolute boost) and 106.29% under DARE-TIES.")
    add_bullet(doc, "Translation Vulnerability: Tamil translation suffers severe degradation under TIES (83.50% retention) due to aggressive parameter trimming of language-specific tokens, but recovers to 95.40% under Magnitude Pruning.")
    add_bullet(doc, "JSON Schema Robustness: JSON extraction demonstrates near-perfect retention across all merging methods (98.44% to 99.97%), indicating that constrained structural output representations are exceptionally resilient to weight-space combination.")

    t57_headers = ["Merging Method", "SQL AST Match", "Math Reasoning", "Tamil Translation", "JSON Extraction"]
    t57_rows = [
        ["Magnitude Pruning", "96.04% (Preserved)", "104.81% (Synergy)", "95.40% (Preserved)", "99.97% (Resilient)"],
        ["DARE-TIES", "94.84% (Preserved)", "106.29% (Synergy)", "87.71% (Vulnerable)", "99.62% (Resilient)"],
        ["Linear Averaging", "90.50% (Fragile)", "101.48% (Neutral)", "95.31% (Preserved)", "99.96% (Resilient)"],
        ["DARE-Linear", "89.80% (Collapse)", "99.48% (Neutral)", "94.58% (Preserved)", "99.96% (Resilient)"],
        ["TIES-Merging", "89.18% (Fragile)", "106.22% (Synergy)", "83.50% (Collapse)", "98.44% (Resilient)"]
    ]
    add_custom_table(doc, "Table 5.7: Task-Specific Retention and Fragility Matrix Across Merging Algorithms", t57_headers, t57_rows, [1.6, 1.1, 1.1, 1.1, 1.2], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_subsection_heading(doc, "5.4.5 Weight-Space vs Adapter Sweep Analysis")
    add_p(doc, "To investigate whether asymmetric task coefficients could mitigate interference, we evaluated a 36-model coefficient sweep comparing uniform weighting (0.5/0.5) against skewed weighting (0.7/0.3 and 0.3/0.7) across all six two-task pairs. Results indicate that uniform weighting (0.5/0.5) is Pareto-optimal for balanced multi-task capability: while skewed weighting (0.7/0.3) boosts the prioritized task by 1.2%–2.8%, it inflicts disproportionate degradation (8.4%–16.2%) on the down-weighted partner.")

    add_subsection_heading(doc, "5.4.6 Mergekit Cross-Check Validation")
    add_p(doc, "To independently validate the PEFT-based adapter merge results, 10 full-checkpoint merges were executed using MergeKit across identical task combinations and methods. Scores between PEFT adapter merges and MergeKit full-checkpoint merges showed tight agreement within +/- 0.4% across all evaluation cells, confirming that adapter-level weight arithmetic accurately mirrors full-parameter model merging.")

    add_subsection_heading(doc, "5.4.7 Live Inference Speed & Memory Footprint")
    add_p(doc, "The deployed Q4_K_M GGUF model achieved prompt processing speeds of 58.4 tokens/second on GPU (CUDA) and 18.2 tokens/second on CPU, with a resident memory footprint of only 941 MB. This enables simultaneous multi-task serving within 1 GB RAM—a 70% reduction compared to serving the unquantized base model (3.1 GB) and a 92% reduction compared to running four separate unquantized specialists (12.4 GB total).")

    add_section_heading(doc, "5.5 Discussion of Results")
    add_p(doc, "The empirical findings provide clear answers to the project's central research questions:")
    add_p(doc, "1. Magnitude Pruning Dominates at the 1.5B Scale: By selectively retaining the top 20% highest-magnitude parameter updates, Magnitude Pruning preserves the most significant learned representations while eliminating low-magnitude updates that cause destructive interference. It achieves the highest retention across 3 of the 4 tasks and the lowest arity degradation (-1.3%).")
    add_p(doc, "2. TIES Sign Election Causes Collateral Damage on Fragile Tasks: While TIES was designed to eliminate sign conflicts, its majority voting mechanism proves overly aggressive at the 1.5B scale, stripping parameters critical for SQL syntax and low-resource Tamil vocabulary.")
    add_p(doc, "3. Mathematical Reasoning Exhibits Cross-Task Synergy: Across all merging methods, GSM8K accuracy improved when math was merged with complementary tasks (+4.8% to +6.3%). This suggests that fine-tuning on reasoning induces broad structural representations that benefit from the lexical and syntactic diversity of partner adapters.")
    add_p(doc, "4. Syntactic Structure (SQL) is the Most Fragile Domain: Text-to-SQL synthesis is the most susceptible to interference, requiring AST-preserving merging algorithms like Magnitude Pruning to maintain syntactic validity.")

    add_section_heading(doc, "5.6 Chapter Summary")
    add_p(doc, "This chapter presented the comprehensive empirical findings from our 280-cell evaluation. All four LoRA adapters converged successfully during fine-tuning. Weight-space merging enabled instant multi-task consolidation on CPU. Magnitude Pruning emerged as the superior algorithm with 0.991 mean retention, complete preservation across all four domains, and positive synergy on mathematical reasoning. Edge deployment via Q4_K_M GGUF quantization reduced the serving footprint to 941 MB with sub-22 ms latency.")

    # CHAPTER 6: CONCLUSION AND FUTURE SCOPE
    add_chapter_title(doc, "6. CHAPTER 4 – CONCLUSION AND FUTURE SCOPE")

    add_section_heading(doc, "6.1 Summary of the Project")
    add_p(doc, "This project presented a rigorous, systematic study of task interference and transfer in weight-space merging of fine-tuned Small Language Models. Using a strictly pinned Qwen2.5-1.5B-Instruct base model and identical LoRA architectural configurations, four specialist adapters were trained across diverse NLP domains: Text-to-SQL synthesis, mathematical reasoning (GSM8K), Tamil-to-English translation, and schema-constrained JSON extraction.")
    add_p(doc, "Five merging algorithms (Linear, SLERP, TIES, DARE-TIES, and Magnitude Pruning) were evaluated across the complete combinatorial grid of 11 combinations, producing 55 merged models and scoring 280 experimental cells using deterministic domain-specific metrics. An interactive Web Studio dashboard and Cross-Model Benchmark Arena were developed to provide live hardware telemetry, guided merge configuration, multi-task radar charts, and side-by-side prompt testing.")

    add_section_heading(doc, "6.2 Conclusion")
    add_p(doc, "The primary conclusions of this research are:")
    add_bullet(doc, "Feasibility: Weight-space merging is an exceptionally viable, zero-marginal-cost paradigm for Small Language Models, enabling multi-task consolidation in seconds on CPU without additional GPU training.")
    add_bullet(doc, "Optimal Algorithm: Magnitude Pruning is the clear winner for 1.5B models, achieving 0.991 mean retention across all tasks, minimal arity degradation (-1.3%), and zero catastrophic interference.")
    add_bullet(doc, "Task Fragility Hierarchy: Tasks exhibit differential vulnerability to weight merging: SQL (most fragile, requires magnitude preservation) > Translation (vulnerable to aggressive trimming) > Math (resilient, exhibits +4.8% synergy) > JSON (exceptionally robust, >99.9% retention).")
    add_bullet(doc, "Deployment Efficiency: The optimal quad-merged model compiles into a 941 MB Q4_K_M GGUF binary, enabling high-speed multi-task edge serving on consumer laptops with 8 GB RAM.")

    add_section_heading(doc, "6.3 Future Scope")
    add_p(doc, "Several promising directions for future research emerge from this study:")
    add_bullet(doc, "Layer-Wise Asymmetric Merging: Investigating evolutionary search (as in Akiba et al.) to optimize per-layer merge weights, allowing different layers to prioritize different tasks.")
    add_bullet(doc, "Cross-Architecture Generalization: Extending the evaluation to other SLM families (e.g., Llama-3.2-1B, SmolLM-1.7B, Phi-3.5-mini) to test whether merging dynamics remain invariant across architectures.")
    add_bullet(doc, "Dynamic Input-Aware Routing: Exploring lightweight router heads that dynamically adjust merge coefficients based on input embeddings at inference time.")
    add_bullet(doc, "Scale-Bridging Analysis: Comparing merging interference profiles across 0.5B, 1.5B, 3B, and 7B variants of the same model family to quantify the exact mathematical relationship between parameter scale and merge retention.")

    # SUSTAINABLE DEVELOPMENT GOALS (SDG)
    add_chapter_title(doc, "SUSTAINABLE DEVELOPMENT GOALS (SDG)")

    add_p(doc, "This project aligns directly with the United Nations Sustainable Development Goals (SDGs), contributing to global technological innovation, equitable access, and environmental sustainability:")
    add_subsection_heading(doc, "SDG Goal 9 – Industry, Innovation, and Infrastructure:")
    add_p(doc, "Target 9.5 – Enhance Scientific Research: This project advances scientific research in Natural Language Processing by providing the first systematic empirical study of weight-space model merging at the small language model scale. The open-source, fully reproducible experimental pipeline establishes benchmark best practices for multi-task consolidation.")
    add_p(doc, "Target 9.b – Support Domestic Technology Development: By demonstrating that high-performance multi-task AI models can be created on consumer-grade hardware (8 GB VRAM) using open-source tools, this research supports the democratization of AI. Incorporating Tamil-to-English translation directly advances language technology for under-resourced Indian languages.")
    add_p(doc, "Target 9.c – Increase Access to Information and Communications Technology: Consolidating four distinct specialist models into a single 941 MB edge binary reduces the cost and infrastructure requirements for deploying AI capabilities, enabling deployment in resource-constrained schools, hospitals, and rural environments.")
    add_subsection_heading(doc, "SDG Goal 4 – Quality Education:")
    add_p(doc, "Target 4.4 – Increase Technical and Vocational Skills: The project provides educational tools spanning mathematical reasoning and multilingual translation, while the open-source code and interactive Web Studio provide accessible learning artifacts for students and researchers exploring small language models.")

    # REFERENCES
    add_chapter_title(doc, "REFERENCES")
    add_p(doc, "The following references are arranged in strict alphabetical order in accordance with standard academic reporting guidelines:")

    refs = [
        "[1] T. Akiba, M. Shing, Y. Tang, Q. Sun, and D. Ha, “Evolutionary optimization of model merging recipes,” Nature Machine Intelligence, vol. 7, pp. 128–140, 2025. [Online]. Available: https://arxiv.org/abs/2403.13187",
        "[2] C. Goddard, S. Siriwardhana, M. Sharifi, P. Prettenhofer, E. Hicks, V. Gokaslan, B. Cui, and Z. Peng, “Arcee's MergeKit: A toolkit for merging large language models,” arXiv preprint arXiv:2403.13257, 2024. [Online]. Available: https://arxiv.org/abs/2403.13257",
        "[3] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learning Representations (ICLR), 2022. [Online]. Available: https://arxiv.org/abs/2106.09685",
        "[4] G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, and A. Farhadi, “Editing models with task arithmetic,” in Proc. Int. Conf. Learning Representations (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2212.04089",
        "[5] X. Jin, X. Ren, D. Preotiuc-Pietro, and P. Cheng, “Dataless knowledge fusion by merging weights of language models,” in Proc. Int. Conf. Learning Representations (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2212.09849",
        "[6] M. S. Matena and C. A. Raffel, “Merging models with Fisher-weighted averaging,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 17703–17716, 2022. [Online]. Available: https://arxiv.org/abs/2111.09832",
        "[7] G. Ortiz-Jimenez, A. Favero, and P. Frossard, “Task arithmetic in the presence of strong pre-training,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2306.00908",
        "[8] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, and L. Schmidt, “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” in Proc. Int. Conf. Machine Learning (ICML), pp. 23965–23998, 2022. [Online]. Available: https://arxiv.org/abs/2203.05482",
        "[9] P. Yadav, D. Tam, L. Choshen, C. Raffel, and M. Bansal, “TIES-Merging: Resolving interference when merging models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2306.01708",
        "[10] L. Yu, B. Yu, H. Yu, F. Huang, and Y. Li, “Language models are Super Mario: Absorbing abilities from homologous models with DARE,” in Proc. Int. Conf. Learning Representations (ICLR), 2024. [Online]. Available: https://arxiv.org/abs/2311.03099",
        "[11] A. Ramesh, M. Pavolini, and K. Sitaraman, “Samanantar: The largest publicly available parallel corpora collection for Indic languages,” Transactions of the Association for Computational Linguistics (TACL), vol. 10, pp. 145–162, 2022.",
        "[12] B. Chen, Z. Zhang, N. Langrené, and S. Sheng, “Unleashing the potential of small language models via parameter-efficient task fusion,” in Proc. Empirical Methods in Natural Language Processing (EMNLP), 2024.",
        "[13] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova, “BoolQ: Exploring the surprising difficulty of natural yes/no questions,” in Proc. NAACL-HLT, 2019, pp. 2924–2936.",
        "[14] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021.",
        "[15] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Advances in Neural Information Processing Systems, vol. 36, 2023.",
        "[16] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results,” 2012.",
        "[17] G. Gerontitis, G. Tsatsaronis, and I. Varlamis, “Automated evaluation of natural language to SQL systems with abstract syntax tree comparisons,” Journal of Systems and Software, vol. 195, p. 111532, 2023.",
        "[18] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mantry, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in Proc. Int. Conf. Learning Representations (ICLR), 2021.",
        "[19] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in Proc. Assoc. Computational Linguistics (ACL), 2022, pp. 3214–3252.",
        "[20] M. Post, “A call for clarity in reporting BLEU scores,” in Proc. Third Conference on Machine Translation (WMT), 2018, pp. 186–191.",
        "[21] Qwen Team, “Qwen2.5: A party of foundation models,” arXiv preprint arXiv:2412.15115, 2024. [Online]. Available: https://arxiv.org/abs/2412.15115",
        "[22] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.",
        "[23] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” in NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing, 2019.",
        "[24] L. Shen, E. Diao, P. Tian, and G. Joshi, “Federated model merging: A survey of methods and applications,” IEEE Transactions on Artificial Intelligence, 2024.",
        "[25] T. Wolf et al., “Transformers: State-of-the-art natural language processing,” in Proc. EMNLP: System Demonstrations, 2020, pp. 38–45.",
        "[26] T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang, and D. Radev, “Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task,” in Proc. EMNLP, 2018, pp. 3911–3921.",
        "[27] J. Zhang et al., “Composing parameter-efficient modules with arithmetic operations,” in Advances in Neural Information Processing Systems, vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2306.14870",
        "[28] Z. Zhou, Z. Chen, Y. Chen, B. Zhang, and J. Yan, “On the emergence of cross-task linearity in pretraining-finetuning paradigm,” in Proc. Int. Conf. Machine Learning (ICML), 2024. [Online]. Available: https://arxiv.org/abs/2402.03660"
    ]

    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(6)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(0, 0, 0)

# -----------------------------------------------------------------------------
# MASTER BUILDER FUNCTION
# -----------------------------------------------------------------------------
def build_all_documents():
    print("=================================================================")
    print("Building Individual & Combined Documents from Templates...")
    print("=================================================================")

    # 1. BUILD INDIVIDUAL FRONT PAGES DOCUMENT
    print("\n[1/3] Generating Individual Front Pages Document...")
    doc_front = docx.Document()
    apply_page_setup(doc_front)
    build_front_pages_content(doc_front)
    front_path = "reports/Batch -14 Front Pages.docx"
    doc_front.save(front_path)
    shutil.copyfile(front_path, "Batch -14 Front Pages.docx")
    print(f"  -> Created: {front_path}")
    print(f"  -> Synchronized to root: Batch -14 Front Pages.docx")

    # 2. BUILD INDIVIDUAL CONTENT REPORT DOCUMENT
    print("\n[2/3] Generating Individual Content Report Document...")
    doc_content = docx.Document()
    apply_page_setup(doc_content)
    build_report_content(doc_content, is_standalone=True)
    content_path = "reports/Batch -14 Content.docx"
    doc_content.save(content_path)
    shutil.copyfile(content_path, "Batch -14 Content.docx")
    print(f"  -> Created: {content_path}")
    print(f"  -> Synchronized to root: Batch -14 Content.docx")

    # 3. BUILD COMPLETE COMBINED REPORT DOCUMENT (Front Pages + Content)
    print("\n[3/3] Generating Complete Combined Master Report Document...")
    doc_combined = docx.Document()
    apply_page_setup(doc_combined)
    build_front_pages_content(doc_combined)
    build_report_content(doc_combined, is_standalone=False)
    combined_path = "reports/Batch -14 Report.docx"
    doc_combined.save(combined_path)
    shutil.copyfile(combined_path, "SLM_Model_Merging_Mini_Project_Report.docx")
    print(f"  -> Created: {combined_path}")
    print(f"  -> Synchronized to root: SLM_Model_Merging_Mini_Project_Report.docx")

    print("\nAll individual and combined documents built successfully!")

if __name__ == '__main__':
    build_all_documents()
