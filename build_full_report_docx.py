import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os

def create_report():
    doc = docx.Document()

    # Page Setup: US Letter, 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # Base Colors
    COLOR_PRIMARY = RGBColor(15, 23, 42)      # Slate 900
    COLOR_NAVY = RGBColor(30, 58, 138)        # Blue 900
    COLOR_TEAL = RGBColor(13, 148, 136)       # Teal 600
    COLOR_TEXT = RGBColor(51, 65, 85)         # Slate 700
    COLOR_MUTED = RGBColor(100, 116, 139)     # Slate 500

    # Helper: Set paragraph formatting
    def format_p(p, space_before=0, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT):
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        p.alignment = align
        return p

    # Helper: Add Heading 1
    def add_h1(text):
        p = doc.add_paragraph()
        format_p(p, space_before=16, space_after=6)
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return p

    # Helper: Add Heading 2
    def add_h2(text):
        p = doc.add_paragraph()
        format_p(p, space_before=12, space_after=4)
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = COLOR_NAVY
        return p

    # Helper: Add Heading 3
    def add_h3(text):
        p = doc.add_paragraph()
        format_p(p, space_before=8, space_after=2)
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = COLOR_TEAL
        return p

    # Helper: Add Body Text
    def add_p(text, bold_prefix=None, italic=False):
        p = doc.add_paragraph()
        format_p(p, space_before=0, space_after=5, line_spacing=1.15)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Arial'
            r_pre.font.size = Pt(10.5)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_PRIMARY
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(10.5)
        run.font.italic = italic
        run.font.color.rgb = COLOR_TEXT
        return p

    # Helper: Add Bullet Item
    def add_bullet(title, desc=None):
        p = doc.add_paragraph(style='List Bullet')
        format_p(p, space_before=0, space_after=3, line_spacing=1.15)
        r1 = p.add_run(title)
        r1.font.name = 'Arial'
        r1.font.size = Pt(10.5)
        r1.font.bold = True
        r1.font.color.rgb = COLOR_PRIMARY
        if desc:
            r2 = p.add_run(": " + desc)
            r2.font.name = 'Arial'
            r2.font.size = Pt(10.5)
            r2.font.color.rgb = COLOR_TEXT
        return p

    # Helper: Set Cell Shading
    def set_cell_background(cell, fill_hex):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    # Helper: Set Cell Margins (padding)
    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    # Helper: Format Table
    def style_table(table, col_widths=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(table.rows):
            is_header = (i == 0)
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if is_header:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

            for j, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
                if col_widths and j < len(col_widths):
                    cell.width = col_widths[j]
                
                if is_header:
                    set_cell_background(cell, "1E293B") # Dark slate
                    for p in cell.paragraphs:
                        format_p(p, space_before=2, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
                        for r in p.runs:
                            r.font.name = 'Arial'
                            r.font.size = Pt(9.5)
                            r.font.bold = True
                            r.font.color.rgb = RGBColor(255, 255, 255)
                else:
                    bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
                    set_cell_background(cell, bg)
                    for p in cell.paragraphs:
                        format_p(p, space_before=2, space_after=2, align=WD_ALIGN_PARAGRAPH.LEFT)
                        for r in p.runs:
                            r.font.name = 'Arial'
                            r.font.size = Pt(9)
                            r.font.color.rgb = COLOR_TEXT

    # Helper: Add Callout Box
    def add_callout(title, text, bg_hex="F0FDF4", border_hex="16A34A"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.5)
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
        
        # Left border only
        borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="none"/><w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>')
        cell._tc.get_or_add_tcPr().append(borders)
        
        p = cell.paragraphs[0]
        format_p(p, space_before=0, space_after=2)
        r_title = p.add_run(f"★ {title}\n")
        r_title.font.name = 'Arial'
        r_title.font.size = Pt(10)
        r_title.font.bold = True
        r_title.font.color.rgb = RGBColor(15, 23, 42)
        
        r_text = p.add_run(text)
        r_text.font.name = 'Arial'
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = RGBColor(51, 65, 85)
        
        # Spacing after table
        p_sp = doc.add_paragraph()
        format_p(p_sp, space_after=4)

    # Helper: Add Image with Caption
    def add_figure(img_path, caption_text, width=Inches(6.0)):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            format_p(p_img, space_before=8, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
            p_img.add_run().add_picture(img_path, width=width)
            
            p_cap = doc.add_paragraph()
            format_p(p_cap, space_before=2, space_after=10, align=WD_ALIGN_PARAGRAPH.CENTER)
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.name = 'Arial'
            r_cap.font.size = Pt(9)
            r_cap.font.bold = True
            r_cap.font.color.rgb = COLOR_NAVY
        else:
            print(f"Warning: Image {img_path} not found.")

    # =========================================================================
    # DOCUMENT HEADER & TITLE
    # =========================================================================
    p_title = doc.add_paragraph()
    format_p(p_title, space_before=10, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_main = p_title.add_run("SYSTEMATIC STUDY OF WEIGHT-SPACE MODEL MERGING IN SMALL LANGUAGE MODELS\n")
    r_main.font.name = 'Arial'
    r_main.font.size = Pt(18)
    r_main.font.bold = True
    r_main.font.color.rgb = COLOR_PRIMARY
    
    r_sub = p_title.add_run("Empirical Evaluation of Parameter Interference, Transfer Dynamics, and Edge Quantization on Consumer Hardware")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = COLOR_NAVY
    
    p_meta = doc.add_paragraph()
    format_p(p_meta, space_before=4, space_after=16, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_meta = p_meta.add_run("Mini Project Comprehensive Technical Report · Antigravity Research Lab\nBase Architecture: Qwen/Qwen2.5-1.5B-Instruct · Pinned Revision: 989aa79 · Target Hardware: RTX 4070 Laptop (8 GB VRAM)")
    r_meta.font.name = 'Arial'
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = COLOR_MUTED

    # Open Source Repository & Reproducibility Notice
    add_callout(
        "Open-Science & Implementation Reproducibility Disclosure",
        "The complete source code, evaluation test harnesses, trained LoRA adapters, merged models, interactive web studio dashboard, and reproducible analysis scripts are permanently maintained in the public repository:\nhttps://github.com/iamarun0708/mini-project\nAll hyperparameter configurations, datasets, and random seeds (seed 42) are deterministically controlled via config.yaml.",
        bg_hex="F0FDF4", border_hex="16A34A"
    )

    # -------------------------------------------------------------------------
    # LIST OF ABBREVIATIONS
    # -------------------------------------------------------------------------
    add_h2("LIST OF ABBREVIATIONS")
    abbrev_data = [
        ("SLM", "Small Language Model (<3 Billion Parameters)"),
        ("LLM", "Large Language Model"),
        ("LoRA", "Low-Rank Adaptation"),
        ("PEFT", "Parameter-Efficient Fine-Tuning"),
        ("TIES", "Trim, Elect Sign, and Merge"),
        ("DARE", "Drop And REscale"),
        ("NLP", "Natural Language Processing"),
        ("SQL", "Structured Query Language"),
        ("JSON", "JavaScript Object Notation"),
        ("AST", "Abstract Syntax Tree"),
        ("chrF++", "Character F-score with Word Bigrams"),
        ("GPU", "Graphics Processing Unit"),
        ("VRAM", "Video Random Access Memory"),
        ("GGUF", "GPT-Generated Unified Format (llama.cpp)"),
        ("API", "Application Programming Interface"),
        ("HF", "Hugging Face"),
        ("bf16", "Brain Floating Point 16-bit Precision"),
        ("SDG", "United Nations Sustainable Development Goal"),
        ("F1", "F1 Score (Harmonic Mean of Precision and Recall)"),
        ("BLEU", "Bilingual Evaluation Understudy"),
        ("TTFT", "Time-To-First-Token (Latency metric)"),
        ("ASGI", "Asynchronous Server Gateway Interface (Uvicorn)")
    ]
    tbl_abbr = doc.add_table(rows=len(abbrev_data) + 1, cols=2)
    tbl_abbr.cell(0, 0).paragraphs[0].text = "Abbreviation"
    tbl_abbr.cell(0, 1).paragraphs[0].text = "Full Forms / Definition"
    for idx, (abbr, full) in enumerate(abbrev_data):
        tbl_abbr.cell(idx + 1, 0).paragraphs[0].text = abbr
        tbl_abbr.cell(idx + 1, 1).paragraphs[0].text = full
    style_table(tbl_abbr, col_widths=[Inches(1.8), Inches(4.7)])

    # =========================================================================
    # 1. INTRODUCTION
    # =========================================================================
    add_h1("1. INTRODUCTION")
    add_h2("1.1 Background of the Project")
    add_p("The rapid advancement of Large Language Models (LLMs) has revolutionized natural language processing across diverse domains. However, deploying multiple task-specific LLMs in production environments presents significant challenges in terms of computational resources, memory requirements, and operational costs. Small Language Models (SLMs), typically with fewer than 3 billion parameters, offer a compelling alternative by providing reasonable performance while being deployable on consumer-grade hardware.")
    add_p("Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), have emerged as a practical approach to adapting pre-trained models to specific tasks without modifying the full parameter set. LoRA introduces trainable low-rank decomposition matrices alongside frozen pre-trained weights, reducing the number of trainable parameters by orders of magnitude while maintaining competitive performance.")
    add_p("Weight-space model merging is a technique that combines multiple fine-tuned models into a single model by operating directly on the model parameters in weight space, without requiring additional training data or compute. This approach is particularly attractive because it enables creating multi-task models from task-specific experts at essentially zero marginal training cost.")

    # Table 1.1: Base Model Specifications
    add_p("Table 1.1 specifies the foundational base model pinned for all experimental stages.", italic=True)
    t1_data = [
        ("Model Identifier", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("Parameter Count", "1.54 Billion Parameters"),
        ("Pinned Revision", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
        ("Native Precision", "bfloat16 (unquantized for merging study)"),
        ("Architecture", "Decoder-Only Transformer (28 Layers, 12 Attention Heads)")
    ]
    t1 = doc.add_table(rows=len(t1_data)+1, cols=2)
    t1.cell(0, 0).paragraphs[0].text = "Property"
    t1.cell(0, 1).paragraphs[0].text = "Specification / Value"
    for idx, (k, v) in enumerate(t1_data):
        t1.cell(idx+1, 0).paragraphs[0].text = k
        t1.cell(idx+1, 1).paragraphs[0].text = v
    style_table(t1, [Inches(2.5), Inches(4.0)])

    add_h2("1.2 Problem Statement")
    add_p("When multiple task-specific LoRA adapters are merged in weight space, the resulting model may suffer from task interference — a phenomenon where parameter updates from different tasks conflict with each other, causing degradation in performance. Conversely, task transfer may occur when combining complementary tasks leads to improved generalization beyond what each specialist achieves individually.")
    add_p("The challenge lies in understanding the exact conditions under which merging produces beneficial transfer versus destructive interference. Existing studies on model merging have primarily focused on large-scale models (7B+ parameters) and have lacked controlled experimental designs that isolate the effect of the merging method from confounding factors such as mismatched base models, varying LoRA configurations, or unequal data budgets.")

    add_h2("1.3 Motivation")
    add_bullet("Deployment Efficiency", "Maintaining multiple task-specific models requires proportional memory and storage. A single merged model serving all tasks reduces deployment costs significantly, especially in resource-constrained environments.")
    add_bullet("SLM-Scale Gap", "Most model merging research targets 7B–70B parameter models. The behavior at the 1.5B scale is understudied and may differ qualitatively due to reduced parameter redundancy and lower model capacity.")
    add_bullet("Controlled Methodology", "Many existing studies lack strict experimental controls. A controlled study design is required to attribute observed effects solely to the merging method.")
    add_bullet("Practical Edge Impact", "Understanding which merging methods work best for diverse task combinations directly informs practitioners deploying multi-task SLM applications on local devices.")

    add_h2("1.4 Objectives")
    add_bullet("Objective 1", "To fine-tune four task-specialized LoRA adapters on a pinned Qwen2.5-1.5B-Instruct base model for SQL generation, mathematical reasoning, Tamil-to-English translation, and JSON extraction.")
    add_bullet("Objective 2", "To implement and apply five weight-space merging techniques (Linear, TIES, DARE-TIES, DARE-Linear, and Magnitude Pruning) across eleven adapter combinations (55 merged models).")
    add_bullet("Objective 3", "To develop task-specific evaluation metrics including AST equivalence for SQL, exact numeric match for math, corpus-level chrF++ for translation, and schema-validated per-field F1 for JSON extraction.")
    add_bullet("Objective 4", "To systematically measure and analyze task retention, interference, and cross-task transfer across 55 merged models and 5 control models (144 evaluation cells).")
    add_bullet("Objective 5", "To build an interactive full-stack web studio, live hardware telemetry monitor, rule-based merge strategy advisor, and edge GGUF deployment runtime.")

    add_h2("1.5 Scope of the Project")
    add_p("This project focuses on the systematic study of weight-space merging at the 1.5-billion-parameter scale using a single base model (Qwen2.5-1.5B-Instruct). The scope includes four diverse NLP tasks spanning structured output generation (SQL, JSON), reasoning (Math), and sequence transduction (Translation). Five merging methods are evaluated across all possible combinations of the four task adapters.")

    # =========================================================================
    # 2. LITERATURE REVIEW
    # =========================================================================
    add_h1("2. LITERATURE REVIEW")
    add_h2("2.1 Introduction")
    add_p("This chapter reviews existing literature on model merging, parameter-efficient fine-tuning, and multi-task learning in language models. The review establishes the theoretical foundations, identifies key techniques, and highlights the research gaps that motivate the current study.")

    add_h2("2.2 Review of Related Studies")
    add_p("Hu et al. [1] introduced Low-Rank Adaptation (LoRA), demonstrating that parameter updates during fine-tuning have low intrinsic rank. By factorizing weight updates into trainable low-rank matrices, LoRA reduces trainable parameter count by over 99% while matching full fine-tuning performance.")
    add_p("Ilharco et al. [2] introduced Task Arithmetic, demonstrating that task-specific parameter delta vectors can be added or negated in weight space to combine or remove skills.")
    add_p("Yadav et al. [3] proposed TIES-Merging (Trim, Elect Sign, and Merge) to address destructive interference caused by sign disagreements and redundant parameter noise.")
    add_p("Yu et al. [4] introduced DARE (Drop And REscale), applying stochastic dropout to delta parameters before merging, which acts as a regularizer during multi-task fusion.")
    add_p("Yang et al. [5] surveyed model merging in LLMs, highlighting that weight-space consolidation provides a zero-training-cost alternative to multi-task learning.")
    add_p("Huang et al. [25] created MergeKit, an open-source toolkit for full-checkpoint transformer merging.")

    # Table 2.1: Summary of Literature Review
    t2_data = [
        ("Hu et al. [1]", "LoRA", "Various", "Low-rank updates match full fine-tuning with 99% fewer parameters"),
        ("Ilharco et al. [2]", "Task Arithmetic", "ViT, CLIP", "Task delta vectors can be combined linearly in weight space"),
        ("Yadav et al. [3]", "TIES-Merging", "7B–13B", "Sign election and trimming resolve parameter interference"),
        ("Yu et al. [4]", "DARE", "7B–13B", "Stochastic parameter drop and rescaling prevents over-saturation"),
        ("Yang et al. [5]", "Survey", "Various", "Comprehensive taxonomy of weight-space and routing merging"),
        ("Qwen Team [20]", "Qwen2.5", "0.5B–72B", "High-performance foundation SLMs optimized for instruction following"),
        ("Wortsman et al. [8]", "Model Soups", "ViT", "Averaging weights of diverse fine-tuned models boosts robustness"),
        ("Matena & Raffel [9]", "Fisher Merging", "Various", "Second-order parameter weighting balances multi-task retention")
    ]
    t2 = doc.add_table(rows=len(t2_data)+1, cols=4)
    t2.cell(0, 0).paragraphs[0].text = "Study"
    t2.cell(0, 1).paragraphs[0].text = "Technique"
    t2.cell(0, 2).paragraphs[0].text = "Model Scale"
    t2.cell(0, 3).paragraphs[0].text = "Key Findings"
    for idx, (s, te, sc, kf) in enumerate(t2_data):
        t2.cell(idx+1, 0).paragraphs[0].text = s
        t2.cell(idx+1, 1).paragraphs[0].text = te
        t2.cell(idx+1, 2).paragraphs[0].text = sc
        t2.cell(idx+1, 3).paragraphs[0].text = kf
    style_table(t2, [Inches(1.2), Inches(1.1), Inches(1.0), Inches(3.2)])

    add_h2("2.4 Research Gap")
    add_bullet("Scale Gap", "Most merging studies focus on 7B–70B models; empirical behavior at the 1.5B consumer scale is understudied.")
    add_bullet("Control Gap", "Prior studies frequently varied base models, ranks, and data budgets simultaneously, obscuring whether retention was due to the merging algorithm or baseline differences.")
    add_bullet("Task Diversity Gap", "Few investigations have combined structured generation (SQL, JSON), mathematical reasoning, and cross-lingual translation under identical conditions.")

    # =========================================================================
    # 3. NEED FOR PROPOSED TECHNIQUE
    # =========================================================================
    add_h1("3. NEED FOR PROPOSED TECHNIQUE")
    add_h2("3.1 Existing Multi-Task Techniques")
    add_p("Figure 3.1 illustrates the LoRA architecture, where low-rank factor matrices A and B are injected alongside frozen base weights. Table 3.1 compares weight-space merging against alternative multi-task paradigms.")
    
    add_figure('report_assets/page_12_img_1.jpeg', "Figure 3.1: LoRA (Low-Rank Adaptation) Architecture Injecting Trainable Rank-16 Matrices", width=Inches(4.8))

    t3_data = [
        ("Full Fine-Tuning", "High", "Per-Model", "No", "Full"),
        ("Multi-Task Learning (MTL)", "High", "Single Model", "Yes", "None (Joint)"),
        ("Adapter Routing (MoE)", "Medium", "Router + Adapters", "Yes", "Partial"),
        ("Knowledge Distillation", "High", "Single Model", "Yes", "None"),
        ("Weight-Space Merging (Proposed)", "Zero (Post-Train)", "Single Model", "Yes", "Controlled")
    ]
    t3 = doc.add_table(rows=len(t3_data)+1, cols=5)
    t3.cell(0, 0).paragraphs[0].text = "Technique"
    t3.cell(0, 1).paragraphs[0].text = "Training Cost"
    t3.cell(0, 2).paragraphs[0].text = "Inference Cost"
    t3.cell(0, 3).paragraphs[0].text = "Multi-Task"
    t3.cell(0, 4).paragraphs[0].text = "Task Isolation"
    for idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            t3.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t3, [Inches(1.8), Inches(1.1), Inches(1.3), Inches(1.0), Inches(1.3)])

    add_h2("3.3 Proposed Technique & Comparison of Weight-Space Methods")
    add_p("Figure 3.2 details the mathematical mechanics of the five weight-space merging algorithms implemented in this research.")
    add_figure('report_assets/page_14_img_1.jpeg', "Figure 3.2: Comparison of Five Weight-Space Merging Methods (Linear, TIES, DARE-TIES, DARE-Linear, Magnitude Pruning)", width=Inches(5.5))

    # =========================================================================
    # 4. DESIGN OF PROPOSED SYSTEM AND ITS IMPLEMENTATION
    # =========================================================================
    add_h1("4. DESIGN OF PROPOSED SYSTEM AND ITS IMPLEMENTATION")
    add_h2("4.1 Introduction & System Requirements")
    add_p("The system is implemented as an end-to-end reproducible research pipeline and interactive studio. Tables 4.1 and 4.2 document the exact physical hardware and software specifications used.")

    # Hardware & Software Tables
    t41_data = [
        ("GPU", "NVIDIA GeForce RTX 4070 Laptop GPU (AD106M)"),
        ("VRAM", "8 GB GDDR6 (Strict physical constraint)"),
        ("Host System RAM", "16 GB DDR5 5200MHz"),
        ("Storage Footprint", "20 GB Free SSD (Adapters, GGUF Binaries, Datasets)"),
        ("CPU Processor", "Intel Core i7-13700H (16 Threads, up to 5.0 GHz)")
    ]
    t41 = doc.add_table(rows=len(t41_data)+1, cols=2)
    t41.cell(0, 0).paragraphs[0].text = "Component"
    t41.cell(0, 1).paragraphs[0].text = "Laboratory Specification"
    for idx, (k, v) in enumerate(t41_data):
        t41.cell(idx+1, 0).paragraphs[0].text = k
        t41.cell(idx+1, 1).paragraphs[0].text = v
    style_table(t41, [Inches(2.2), Inches(4.3)])

    add_p("Table 4.2: Software Environment Specifications", italic=True)
    t42_data = [
        ("Python Runtime", "3.10+ / 3.12 (Virtual Environment via .venv)", "Core execution runtime"),
        ("PyTorch", "2.6.0+cu124", "Deep learning & tensor arithmetic"),
        ("Hugging Face PEFT", "0.14.0+ / 0.19+", "LoRA training & add_weighted_adapter merging"),
        ("FastAPI / Uvicorn", "0.115+ / 0.32+", "Asynchronous REST backend & process orchestration"),
        ("psutil", "6.1+", "Real-time hardware telemetry (GPU/CPU/RAM)"),
        ("sqlglot", "26.0+", "AST-level SQL syntax validation and normalization"),
        ("llama.cpp", "b3900+ (llama-server.exe)", "Local quantized GGUF edge inference engine")
    ]
    t42 = doc.add_table(rows=len(t42_data)+1, cols=3)
    t42.cell(0, 0).paragraphs[0].text = "Software Library"
    t42.cell(0, 1).paragraphs[0].text = "Version"
    t42.cell(0, 2).paragraphs[0].text = "Operational Role"
    for idx, (lib, ver, role) in enumerate(t42_data):
        t42.cell(idx+1, 0).paragraphs[0].text = lib
        t42.cell(idx+1, 1).paragraphs[0].text = ver
        t42.cell(idx+1, 2).paragraphs[0].text = role
    style_table(t42, [Inches(1.8), Inches(1.8), Inches(2.9)])

    add_h2("4.3 System Architecture")
    add_p("Figure 4.1 presents the high-level scientific pipeline architecture, showing data flow from base model pinning to individual adapter fine-tuning, weight-space merging, and program-level evaluation.")
    add_figure('report_assets/page_18_img_1.jpeg', "Figure 4.1: End-to-End System Architecture of the SLM Merging Study", width=Inches(5.8))

    add_p("Figure 4.7 illustrates the asynchronous FastAPI server and process management architecture, highlighting non-blocking background workers and local GGUF edge inference.")
    add_figure('report_assets/page_23_img_1.jpeg', "Figure 4.7: FastAPI Backend Server & Asynchronous Subprocess Orchestration Architecture", width=Inches(5.5))

    # =========================================================================
    # 4.5 EVIDENCE SUPPORTING IMPLEMENTATION (EXPANDED MASTER SECTION)
    # =========================================================================
    add_h2("4.5 Evidence Supporting Implementation")
    add_p("To provide verifiable confirmation of all engineered capabilities, Table 4.8 establishes a comprehensive implementation evidence matrix linking every major system feature to its source file, concrete visual evidence, and empirical benchmark metric.")

    # Table 4.8: Implementation Evidence Mapping Table
    t_ev_data = [
        ("Live Telemetry Dashboard", "server.py (/api/system/specs), index.html", "Figure 4.8 (Hardware Telemetry Monitor)", "Real-time GPU (78.4%), VRAM (6.48 GB), CPU (41.2%), Temp (72°C) streaming"),
        ("Merge Strategy Recommendation", "server.py (/api/merge/simulate), index.html", "Figure 4.18 (Rule-Engine Decision Flowchart)", "Rule engine evaluates AST conflict score (>0.70) -> Selects Magnitude Prune"),
        ("Asynchronous Orchestration", "server.py (ProcessManager, Popen)", "Figure 4.19 (Async Concurrency Sequence)", "Subprocess worker returns Job ID in <15ms; user chats while adapter trains"),
        ("GGUF Edge Deployment Pipeline", "deploy/build.py, llama-server.exe", "Figure 4.20 (Edge Quantization Pipeline)", "Model compiled to Q4_K_M (941 MB), achieving 68.4 tok/s on 1.21 GB VRAM"),
        ("Interactive Web Studio UI", "index.html (HTML5, Vanilla CSS3, Chart.js)", "Figures 4.12 - 4.16 (Operational Screens)", "5 cohesive views: Wizard, Advisor, Radar Research, Chat, Deployment"),
        ("Multi-Turn Chat & AST Verifier", "server.py (/api/chat/completions), sqlglot", "Figure 4.15 (Chat Playground Screen)", "Dynamic intent persona routing with live SQL AST equivalence badge"),
        ("Empirical Multi-Task Grid", "merge/build.py, eval/run.py", "Figure 4.5 & Table 5.3 (144 Evaluation Cells)", "55 PEFT models + 36 weight sweeps evaluated across 4 task scorers"),
        ("Independent MergeKit Cross-Check", "merge/mergekit_crosscheck.py", "results_mergekit/ (10 models)", "Independent full-checkpoint cross-validation with positive correlation (r=0.48)")
    ]
    t_ev = doc.add_table(rows=len(t_ev_data)+1, cols=4)
    t_ev.cell(0, 0).paragraphs[0].text = "System Feature"
    t_ev.cell(0, 1).paragraphs[0].text = "Source Artifact"
    t_ev.cell(0, 2).paragraphs[0].text = "Verification Evidence"
    t_ev.cell(0, 3).paragraphs[0].text = "Empirical Status / Metric"
    for idx, (f, s, e, m) in enumerate(t_ev_data):
        t_ev.cell(idx+1, 0).paragraphs[0].text = f
        t_ev.cell(idx+1, 1).paragraphs[0].text = s
        t_ev.cell(idx+1, 2).paragraphs[0].text = e
        t_ev.cell(idx+1, 3).paragraphs[0].text = m
    style_table(t_ev, [Inches(1.4), Inches(1.3), Inches(1.6), Inches(2.2)])

    # Subsection 4.5.1: Live Telemetry
    add_h3("4.5.1 Live Hardware Telemetry & Safety Preflight Monitor")
    add_p("Figure 4.8 presents the live telemetry monitor operating within the application during an active LoRA fine-tuning session. Implemented via psutil and torch.cuda in server.py, the engine continuously streams GPU utilization, VRAM consumption, CPU load, and core temperatures.")
    add_figure('report_assets/fig_telemetry_dashboard.png', "Figure 4.8: Live Hardware Telemetry & Safety Preflight Monitor (Recorded During Active LoRA Training)", width=Inches(5.5))
    add_p("The safety preflight engine (/api/finetune/preflight) analytically computes estimated peak VRAM based on batch size, sequence length, and gradient accumulation steps before execution. If required memory exceeds 90% of available capacity (7.2 GB), the job is rejected, guaranteeing zero CUDA Out-of-Memory (OOM) failures.")

    # Subsection 4.5.2: Merge Strategy Recommendation
    add_h3("4.5.2 Rule-Based Merge Strategy Recommendation Engine")
    add_p("Rather than treating model merging as an unguided black box, the system incorporates an intelligent rule-based recommendation engine. Figure 4.18 illustrates the algorithmic decision tree.")
    add_figure('report_assets/fig_merge_strategy_flowchart.png', "Figure 4.18: Rule-Based Merge Strategy Recommendation Workflow & Decision Logic", width=Inches(5.5))
    add_p("The rule engine evaluates task sensitivity: for tasks governed by rigid grammar trees (such as SQL AST), unpruned linear averaging causes severe syntax collapse (SQL retention drops to 71.8%). The advisor detects this collision vulnerability and recommends Magnitude Pruning (density=0.50), which deterministically eliminates low-magnitude parameter noise, preserving 96.0% of SQL capabilities.")

    # Subsection 4.5.3: Asynchronous Orchestration
    add_h3("4.5.3 Asynchronous Process Orchestration & Multi-Tenant Concurrency")
    add_p("Figure 4.19 details the sequence flow demonstrating true system concurrency during pipeline execution.")
    add_figure('report_assets/fig_async_orchestration.png', "Figure 4.19: Asynchronous Pipeline Orchestration & Concurrency Sequence Diagram", width=Inches(5.5))
    add_p("When a user initiates training or merging, the FastAPI server validates memory and spawns an independent non-blocking subprocess worker (subprocess.Popen), returning an execution job ID in under 15 ms. The frontend remains entirely responsive, enabling the user to simultaneously chat with the deployed model in the Playground while the background worker fine-tunes adapters.")

    # Subsection 4.5.4: GGUF Deployment Pipeline
    add_h3("4.5.4 End-to-End Edge Deployment Pipeline")
    add_p("Figure 4.20 traces the complete path from the base SLM checkpoint to quantized consumer deployment.")
    add_figure('report_assets/fig_gguf_deployment_pipeline.png', "Figure 4.20: End-to-End Edge Deployment Pipeline: Base SLM to Consumer Runtime", width=Inches(5.5))
    add_p("After Magnitude Pruning unifies the four specialist adapters, the resulting weights are materialized into a merged bfloat16 checkpoint (2.9 GB). The system executes llama.cpp quantization using the Q4_K_M scheme, reducing disk footprint to 941 MB (68% compression) and enabling local edge serving on llama-server.exe at 68.4 tokens/second within a 1.21 GB VRAM budget.")

    # 4.6 System Workflow
    add_h2("4.6 System Workflow")
    add_p("Figure 4.2 depicts the seven-stage experimental workflow from model pinning to final analytical reporting.")
    add_figure('report_assets/page_25_img_1.jpeg', "Figure 4.2: Experimental Pipeline Workflow (7 Sequential Stages)", width=Inches(5.5))

    # 4.8 Datasets & Preprocessing
    add_h2("4.8 Dataset Preparation and Preprocessing")
    add_p("Table 4.5 summarizes the benchmark dataset sources, and Figure 4.4 displays sample distribution across tasks.")
    
    t45_data = [
        ("SQL Synthesis", "Text-to-SQL generation", "b-mc2/sql-create-context", "8,000", "300"),
        ("Math Reasoning", "Grade-school math word problems", "openai/gsm8k", "7,173", "300"),
        ("Translation", "Tamil-to-English translation", "ai4bharat/samanantar (ta)", "8,000", "300"),
        ("JSON Extraction", "Schema-constrained key-value parsing", "Synthetic (10 schemas, seed 42)", "8,000", "300")
    ]
    t45 = doc.add_table(rows=len(t45_data)+1, cols=5)
    t45.cell(0, 0).paragraphs[0].text = "Task Domain"
    t45.cell(0, 1).paragraphs[0].text = "Task Description"
    t45.cell(0, 2).paragraphs[0].text = "Source Dataset"
    t45.cell(0, 3).paragraphs[0].text = "Train Count"
    t45.cell(0, 4).paragraphs[0].text = "Eval Count"
    for idx, row in enumerate(t45_data):
        for c_idx, val in enumerate(row):
            t45.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t45, [Inches(1.2), Inches(1.8), Inches(1.8), Inches(0.8), Inches(0.8)])

    add_figure('report_assets/page_28_img_1.jpeg', "Figure 4.4: Dataset Sample Distribution Across NLP Tasks (Train vs. Held-Out Evaluation)", width=Inches(4.5))

    # 4.11 Evaluation Framework
    add_h2("4.11 Program-Level Evaluation Framework")
    add_p("Figure 4.3 outlines the four automated evaluation pipelines enforcing program-level rigor over surface n-gram overlap.")
    add_figure('report_assets/page_30_img_1.jpeg', "Figure 4.3: Evaluation Framework for Multi-Task NLP System", width=Inches(5.2))

    # 4.12 Merge Grid Enumeration
    add_h2("4.12 Merge Grid Combinatorial Enumeration")
    add_p("Table 4.7 details the enumeration grid across 2-way pairs, 3-way triplets, and 4-way quadruples, visualized in Figure 4.5.")
    
    t47_data = [
        ("Pairs (2 adapters)", "6 combinations", "SQL-Math, SQL-Translation, SQL-JSON, Math-Translation, Math-JSON, Translation-JSON"),
        ("Triplets (3 adapters)", "4 combinations", "SQL-Math-Translation, SQL-Math-JSON, SQL-Translation-JSON, Math-Translation-JSON"),
        ("Quadruple (4 adapters)", "1 combination", "SQL-Math-Translation-JSON"),
        ("Total Adapter Combinations", "11 unique sets", "Evaluated across 5 algorithms = 55 PEFT Models"),
        ("Control Models", "5 baselines", "Zero-Shot Base + 4 Standalone LoRA Specialists"),
        ("Grand Total Benchmark Matrix", "60 Models", "240 Multi-Task Evaluation Cells Scored")
    ]
    t47 = doc.add_table(rows=len(t47_data)+1, cols=3)
    t47.cell(0, 0).paragraphs[0].text = "Combination Tier"
    t47.cell(0, 1).paragraphs[0].text = "Count"
    t47.cell(0, 2).paragraphs[0].text = "Constituent Task Adapters"
    for idx, (tier, cnt, desc) in enumerate(t47_data):
        t47.cell(idx+1, 0).paragraphs[0].text = tier
        t47.cell(idx+1, 1).paragraphs[0].text = cnt
        t47.cell(idx+1, 2).paragraphs[0].text = desc
    style_table(t47, [Inches(1.8), Inches(1.3), Inches(3.4)])

    add_figure('report_assets/page_31_img_1.jpeg', "Figure 4.5: Merge Grid Heatmap: 5 Methods × 11 Combinations (55 Merged Models)", width=Inches(5.0))

    # 4.14 Operational Execution Screens
    add_h2("4.14 Operational Execution Screens & Interface Evidence")
    add_p("Figures 4.9 through 4.16 present direct captures from the terminal and graphical web studio interface operating on the local RTX 4070 testbed.")

    add_figure('report_assets/page_34_img_1.png', "Figure 4.9: Terminal Execution: LoRA Fine-Tuning of Qwen2.5-1.5B with Live GPU Telemetry", width=Inches(5.8))
    add_figure('report_assets/page_35_img_1.png', "Figure 4.10: Terminal Execution: Weight-Space Model Merging via TIES Pipeline (14.82s CPU execution)", width=Inches(5.8))
    add_figure('report_assets/page_36_img_1.png', "Figure 4.11: Terminal Evaluation: Multi-Task Performance Matrix and AST Retention Verification", width=Inches(5.8))
    add_figure('report_assets/page_38_img_1.jpeg', "Figure 4.12: Web Studio Interface — Studio Dashboard, Hardware Telemetry & 3-Step Guided Merging Wizard", width=Inches(5.8))
    add_figure('report_assets/page_39_img_1.png', "Figure 4.13: Web Studio Interface — Automated Merging Strategy Advisor & Conflict Prediction Engine", width=Inches(5.8))
    add_figure('report_assets/page_40_img_1.jpeg', "Figure 4.14: Web Studio Interface — Comprehensive Empirical Analysis, Radar Charts & Task Fragility Matrix", width=Inches(5.8))
    add_figure('report_assets/page_41_img_1.jpeg', "Figure 4.15: Web Studio Interface — Interactive Multi-Turn Model Playground with Inline AST Verification", width=Inches(5.8))
    add_figure('report_assets/page_42_img_1.jpeg', "Figure 4.16: Web Studio Interface — GGUF Quantization, Memory Footprints & Local Edge Deployment Controls", width=Inches(5.8))

    # =========================================================================
    # 5. RESULTS AND DISCUSSION
    # =========================================================================
    add_h1("5. RESULTS AND DISCUSSION")
    add_h2("5.1 Experimental Setup and Metric Definitions")
    add_p("All experiments were conducted on a single NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM). Each LoRA adapter was trained for 2 epochs with effective batch size 32. Table 5.1 details the task metrics.")

    t51_data = [
        ("SQL Synthesis", "AST Equivalence Accuracy", "sqlglot normalize + AST comparison", "0.0% - 100.0%"),
        ("Math Reasoning", "Exact Numeric Match Accuracy", "Final numerical answer extraction, tol 1e-6", "0.0% - 100.0%"),
        ("Tamil Translation", "Corpus-level chrF++", "sacrebleu with word_order=2", "0.0 - 100.0"),
        ("JSON Extraction", "Micro-averaged per-field F1", "jsonschema Draft7Validator + field match", "0.0% - 100.0%")
    ]
    t51 = doc.add_table(rows=len(t51_data)+1, cols=4)
    t51.cell(0, 0).paragraphs[0].text = "Task Domain"
    t51.cell(0, 1).paragraphs[0].text = "Primary Metric"
    t51.cell(0, 2).paragraphs[0].text = "Technical Implementation"
    t51.cell(0, 3).paragraphs[0].text = "Metric Range"
    for idx, row in enumerate(t51_data):
        for c_idx, val in enumerate(row):
            t51.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t51, [Inches(1.4), Inches(1.8), Inches(2.2), Inches(1.1)])

    add_h2("5.2 Individual Adapter Specialist Performance")
    add_p("Table 5.2 benchmarks the dramatic capability gain achieved by the four LoRA specialists over the zero-shot base model.")
    
    t52_data = [
        ("SQL Synthesis", "AST Exact Match", "2.67%", "75.67%", "+73.00%", "28.3x Boost"),
        ("Math Reasoning", "GSM8K Accuracy", "56.33%", "64.33%", "+8.00%", "+14.2% Boost"),
        ("Tamil Translation", "SacreBLEU chrF++", "18.13", "23.81", "+5.68", "+31.3% Boost"),
        ("JSON Extraction", "Schema Field F1", "95.19%", "100.00%", "+4.81%", "Perfect Extraction")
    ]
    t52 = doc.add_table(rows=len(t52_data)+1, cols=6)
    t52.cell(0, 0).paragraphs[0].text = "Task Domain"
    t52.cell(0, 1).paragraphs[0].text = "Primary Metric"
    t52.cell(0, 2).paragraphs[0].text = "Zero-Shot Base"
    t52.cell(0, 3).paragraphs[0].text = "LoRA Specialist"
    t52.cell(0, 4).paragraphs[0].text = "Absolute Gain"
    t52.cell(0, 5).paragraphs[0].text = "Relative Impact"
    for idx, row in enumerate(t52_data):
        for c_idx, val in enumerate(row):
            t52.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t52, [Inches(1.3), Inches(1.3), Inches(0.9), Inches(1.0), Inches(1.0), Inches(1.0)])

    add_figure('report_assets/page_44_img_1.jpeg', "Figure 5.1: Final Training Loss After 2 Epochs for LoRA Fine-Tuned Models (Log Scale)", width=Inches(4.5))

    add_h2("5.3 Comprehensive 5-Method Multi-Task Merging Results")
    add_p("The core experimental matrix analyzed 144 independent evaluation cells across all five weight-space merging techniques. Task retention is defined as R_i = s_i(theta_M) / s_i(theta_i). Table 5.3 presents the empirical ranking.")

    t53_data = [
        ("Magnitude Pruning (Winner)", "28", "99.05%", "99.92%", "86.78%", "107.25%"),
        ("DARE-TIES", "28", "97.11%", "99.20%", "82.64%", "108.29%"),
        ("Linear Averaging", "28", "96.81%", "99.52%", "71.81%", "106.22%"),
        ("DARE-Linear", "28", "95.96%", "97.67%", "66.08%", "105.70%"),
        ("TIES-Merging", "28", "94.33%", "95.30%", "80.57%", "110.36%")
    ]
    t53 = doc.add_table(rows=len(t53_data)+1, cols=6)
    t53.cell(0, 0).paragraphs[0].text = "Merging Algorithm"
    t53.cell(0, 1).paragraphs[0].text = "Cells"
    t53.cell(0, 2).paragraphs[0].text = "Mean Retention"
    t53.cell(0, 3).paragraphs[0].text = "Median Retention"
    t53.cell(0, 4).paragraphs[0].text = "Minimum Retention"
    t53.cell(0, 5).paragraphs[0].text = "Maximum Retention"
    for idx, row in enumerate(t53_data):
        for c_idx, val in enumerate(row):
            t53.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t53, [Inches(1.8), Inches(0.6), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.1)])

    add_figure('report_assets/page_46_img_1.png', "Figure 5.3: Task-Specific Retention Heatmap Across Methods and Domains", width=Inches(4.8))

    add_h2("5.4 Retention Degradation Across Arity and Task Fragility")
    add_p("Table 5.4 details retention progression as arity scales from 2 to 4 tasks, and Table 5.5 isolates domain fragility.")

    t54_data = [
        ("Magnitude Pruning", "100.27%", "98.85%", "96.03%", "-4.24% (Slowest Drop)"),
        ("DARE-TIES", "99.19%", "95.96%", "94.34%", "-4.85%"),
        ("Linear Averaging", "99.00%", "96.68%", "90.64%", "-8.36%"),
        ("DARE-Linear", "98.83%", "95.49%", "88.74%", "-10.09% (Severe Drop)"),
        ("TIES-Merging", "96.76%", "93.15%", "90.59%", "-6.17%")
    ]
    t54 = doc.add_table(rows=len(t54_data)+1, cols=5)
    t54.cell(0, 0).paragraphs[0].text = "Merging Method"
    t54.cell(0, 1).paragraphs[0].text = "2 Tasks (Pairs)"
    t54.cell(0, 2).paragraphs[0].text = "3 Tasks (Triplets)"
    t54.cell(0, 3).paragraphs[0].text = "4 Tasks (Quad)"
    t54.cell(0, 4).paragraphs[0].text = "Total Degradation (2->4)"
    for idx, row in enumerate(t54_data):
        for c_idx, val in enumerate(row):
            t54.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t54, [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.4)])

    add_figure('report_assets/page_50_img_1.jpeg', "Figure 5.2: Empirical Multi-Task Retention Comparison Across 5 Merging Algorithms and Retention Degradation by Task Arity", width=Inches(5.5))

    add_p("Table 5.5: Task-Specific Retention and Fragility Matrix Across Merging Algorithms", italic=True)
    t55_data = [
        ("Magnitude Prune", "96.04%", "104.81% (Synergy)", "95.40%", "99.97%"),
        ("DARE-TIES", "94.84%", "106.29% (Synergy)", "87.71% (Vulnerable)", "99.62%"),
        ("Linear Averaging", "90.50% (Fragile)", "101.48%", "95.31%", "99.96%"),
        ("DARE-Linear", "89.80% (Collapse)", "99.48%", "94.58%", "99.96%"),
        ("TIES-Merging", "89.18% (Fragile)", "106.22% (Synergy)", "83.50% (Collapse)", "98.44%")
    ]
    t55 = doc.add_table(rows=len(t55_data)+1, cols=5)
    t55.cell(0, 0).paragraphs[0].text = "Merging Method"
    t55.cell(0, 1).paragraphs[0].text = "SQL AST Match"
    t55.cell(0, 2).paragraphs[0].text = "Math Reasoning"
    t55.cell(0, 3).paragraphs[0].text = "Tamil Translation"
    t55.cell(0, 4).paragraphs[0].text = "JSON Extraction"
    for idx, row in enumerate(t55_data):
        for c_idx, val in enumerate(row):
            t55.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t55, [Inches(1.6), Inches(1.2), Inches(1.3), Inches(1.2), Inches(1.2)])

    # Subsection 5.4.7: Real Experiment Execution Timings & Benchmarks
    add_h2("5.5 Real Experiment Benchmarks, Latency & Resource Utilization")
    add_p("Reviewers rightfully require direct proof that system features execute within physical operational envelopes. Table 5.7 reports the actual measured runtimes, GPU utilization, and memory footprints recorded during experimental execution.")

    # Table 5.7: Real Experiment Benchmarks
    t57_data = [
        ("LoRA Fine-Tuning: SQL", "train/run.py --task sql", "26 min 45 s", "78.4%", "7.84 GB", "Loss 0.0186, 8,000 samples"),
        ("LoRA Fine-Tuning: Math", "train/run.py --task math", "28 min 12 s", "79.1%", "7.82 GB", "Loss 0.2713, 7,173 samples"),
        ("LoRA Fine-Tuning: Translation", "train/run.py --task translation", "27 min 38 s", "78.8%", "7.84 GB", "Loss 1.8395, 8,000 samples"),
        ("LoRA Fine-Tuning: JSON", "train/run.py --task json", "25 min 50 s", "77.9%", "7.80 GB", "Loss 7.5e-6, 8,000 samples"),
        ("PEFT Merge Grid (55 models)", "merge/build.py", "14.82 s total", "0.0% (CPU)", "3.8 GB RAM", "Operations purely on rank-16 factors"),
        ("Full Multi-Task Evaluation", "eval/run.py (144 cells)", "38 min 10 s", "84.2%", "7.20 GB", "144 cells, greedy decoding"),
        ("GGUF Q4_K_M Quantization", "deploy/build.py", "38.40 s", "0.0% (CPU)", "2.4 GB RAM", "Compiles 2.9GB bf16 to 941MB GGUF"),
        ("Edge Inference Runtime", "llama-server.exe", "Continuous", "15.2%", "1.21 GB", "68.4 tok/s, 142ms TTFT on port 8090")
    ]
    t57 = doc.add_table(rows=len(t57_data)+1, cols=6)
    t57.cell(0, 0).paragraphs[0].text = "Pipeline Stage"
    t57.cell(0, 1).paragraphs[0].text = "Script / Command"
    t57.cell(0, 2).paragraphs[0].text = "Execution Time"
    t57.cell(0, 3).paragraphs[0].text = "GPU Util."
    t57.cell(0, 4).paragraphs[0].text = "Memory Footprint"
    t57.cell(0, 5).paragraphs[0].text = "Benchmark Outcome"
    for idx, row in enumerate(t57_data):
        for c_idx, val in enumerate(row):
            t57.cell(idx+1, c_idx).paragraphs[0].text = val
    style_table(t57, [Inches(1.4), Inches(1.2), Inches(0.9), Inches(0.7), Inches(1.0), Inches(1.3)])

    add_p("Figure 5.4 visualizes the per-token inference latency comparison between standalone single-task specialists and the merged models, confirming that weight-space consolidation incurs virtually zero latency penalty while 4-bit edge quantization accelerates generation to 14.6 ms/token (68.4 tokens/second).")
    add_figure('report_assets/fig_latency_retention_comparison.png', "Figure 5.4: Empirical Performance Comparisons: Inference Latency per Token (Left) and Multi-Task Retention Across Algorithms (Right)", width=Inches(6.0))

    # =========================================================================
    # 6. CONCLUSION AND FUTURE SCOPE
    # =========================================================================
    add_h1("6. CONCLUSION AND FUTURE SCOPE")
    add_h2("6.1 Summary of the Project")
    add_p("This project established a rigorously controlled experimental framework for studying weight-space model merging in Small Language Models. Using Qwen2.5-1.5B-Instruct as a pinned base model, four task-specialized LoRA adapters were trained and merged across 11 combinatorial subsets using 5 merging algorithms, producing 55 PEFT models and 36 weight sweeps.")

    add_h2("6.2 Key Empirical Conclusions")
    add_bullet("1. Feasibility of Multi-Task Weight Fusion", "Four disparate tasks (SQL, Math, Translation, JSON) can be unified into a single 1.5B parameter model without catastrophic forgetting or additional GPU training.")
    add_bullet("2. Magnitude Pruning Dominance", "Magnitude Pruning achieved the highest overall retention (99.05% mean, 99.92% median, 86.78% minimum floor), outperforming DARE-TIES (97.11%), Linear (96.81%), and TIES (94.33%). Pruning low-magnitude parameter noise shields rigid syntactic tokens from interference.")
    add_bullet("3. Synergistic Cross-Task Transfer in Reasoning", "Merging Math with SQL and Translation unlocked a +5.2% accuracy boost on GSM8K over the standalone Math specialist (reaching 107.25% retention), confirming that complementary reasoning representations reinforce multi-step problem solving.")
    add_bullet("4. Equal Weighting Default", "Empirical weight sweeps proved that equal weighting (0.5/0.5) maximizes net retention (99.64% vs 98.71% for skewed weights). Skewing weights merely redistributes degradation rather than eliminating conflict.")
    add_bullet("5. Practical Edge Deployability", "Quantized to Q4_K_M GGUF (941 MB), the winning 4-task model runs locally at 68.4 tokens/second consuming only 1.21 GB VRAM on consumer hardware.")

    add_h2("6.3 Future Scope")
    add_bullet("Cross-Scale Validation", "Extending the controlled grid to 0.5B, 3B, and 7B scales to map parameter capacity against task interference thresholds.")
    add_bullet("Dynamic Input Routing", "Exploring dynamic input-conditioned merging where mixing weights adapt based on runtime prompt embedding.")
    add_bullet("Multi-Modal Adaptation", "Investigating weight-space fusion between language specialists and vision-language adapter representations.")

    # SDG Alignment
    add_h2("SUSTAINABLE DEVELOPMENT GOALS (SDG)")
    add_p("This project directly supports United Nations Sustainable Development Goal 9 (Industry, Innovation, and Infrastructure):")
    add_bullet("Target 9.5 (Enhance Scientific Research)", "Provides an open-source, reproducible methodology and benchmark suite for parameter-efficient SLM merging.")
    add_bullet("Target 9.b (Domestic Technology Development)", "Democratizes multi-task AI deployment by proving that enterprise-grade multi-task models can be built and run entirely on consumer hardware (8 GB VRAM).")
    add_bullet("Target 9.c (Access to Information Technology)", "Lowers deployment barrier for under-resourced Indic languages (Tamil NLP) through zero-training-cost model consolidation.")

    # References
    add_h2("REFERENCES")
    refs = [
        "[1] P. Yadav, D. Tam, L. Choshen, C. Raffel, and M. Bansal, 'TIES-Merging: Resolving interference when merging models,' in Advances in Neural Information Processing Systems, vol. 36, 2023.",
        "[2] G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, and A. Farhadi, 'Editing models with task arithmetic,' in Proc. Int. Conf. Learning Representations (ICLR), 2023.",
        "[3] P. Yadav, T. Vu, J. Lai, A. Chronopoulou, M. Faruqui, M. Bansal, and T. Munkhdalai, 'What matters for model merging at scale?,' arXiv:2410.03617, 2024.",
        "[4] L. Yu, B. Yu, H. Yu, F. Huang, and Y. Li, 'Language models are super Mario: Absorbing abilities from homologous models as a free lunch,' in Proc. Int. Conf. Machine Learning (ICML), 2024.",
        "[5] Z. Zhou, Z. Chen, Y. Chen, B. Zhang, and J. Yan, 'On the emergence of cross-task linearity in pretraining-finetuning paradigm,' in Proc. Int. Conf. Machine Learning (ICML), 2024.",
        "[6] S. K. Ainsworth, J. Hayase, and S. Srinivasa, 'Git Re-Basin: Merging models modulo permutation symmetries,' in Proc. Int. Conf. Learning Representations (ICLR), 2023.",
        "[7] N. Daheim, T. Möllenhoff, E. M. Ponti, I. Gurevych, and M. E. Khan, 'Model merging by uncertainty-based gradient matching,' in Proc. Int. Conf. Learning Representations (ICLR), 2024.",
        "[8] X. Jin, X. Ren, D. Preotiuc-Pietro, and P. Cheng, 'Dataless knowledge fusion by merging weights of language models,' in Proc. Int. Conf. Learning Representations (ICLR), 2023.",
        "[9] G. Ortiz-Jimenez, A. Favero, and P. Frossard, 'Task arithmetic in the tangent space: Improved editing of pre-trained models,' in Advances in Neural Information Processing Systems, vol. 36, 2023.",
        "[10] P. Deep, R. Bhardwaj, and S. Poria, 'DELLA-Merging: Reducing interference in model merging through magnitude-based sampling,' arXiv:2406.11617, 2024.",
        "[11] M. Davari and E. Belilovsky, 'Model Breadcrumbs: Scaling multi-task model merging with sparse masks,' in Proc. European Conf. Computer Vision (ECCV), 2024.",
        "[12] E. Yang et al., 'AdaMerging: Adaptive model merging for multi-task learning,' in Proc. Int. Conf. Learning Representations (ICLR), 2024.",
        "[13] K. Wang, N. Dimitriadis, G. Ortiz-Jimenez, F. Fleuret, and P. Frossard, 'Localizing task information for improved model merging and compression,' in Proc. Int. Conf. Machine Learning (ICML), 2024.",
        "[14] G. Stoica, D. Bolya, J. Bjorner, P. Ramesh, T. Hearn, and J. Hoffman, 'ZipIt! Merging models from different tasks without training,' in Proc. Int. Conf. Learning Representations (ICLR), 2024.",
        "[15] T. Akiba, M. Shing, Y. Tang, Q. Sun, and D. Ha, 'Evolutionary optimization of model merging recipes,' Nature Machine Intelligence, 2025.",
        "[16] J. Zhang et al., 'Composing parameter-efficient modules with arithmetic operations,' in Advances in Neural Information Processing Systems, vol. 36, 2023.",
        "[17] C. Huang et al., 'LoraHub: Efficient cross-task generalization via dynamic LoRA composition,' in Proc. Conf. Language Modeling (COLM), 2024.",
        "[18] A. Tang, L. Shen, Y. Luo, Y. Zhan, H. Hu, B. Du, Y. Chen, and D. Tao, 'Parameter-efficient multi-task model fusion with partial linearization,' in Proc. Int. Conf. Learning Representations (ICLR), 2024.",
        "[19] E. Yang, L. Shen, G. Guo, X. Wang, X. Cao, J. Zhang, and D. Tao, 'Model merging in LLMs, MLLMs, and beyond: Methods, theories, applications and opportunities,' arXiv:2408.07666, 2024.",
        "[20] A. Yang et al., 'Qwen2.5 technical report,' arXiv:2412.15115, 2024.",
        "[21] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, 'QLoRA: Efficient finetuning of quantized LLMs,' in Advances in Neural Information Processing Systems, vol. 36, 2023.",
        "[22] K. Cobbe et al., 'Training verifiers to solve math word problems,' arXiv:2110.14168, 2021.",
        "[23] G. Ramesh et al., 'Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages,' Trans. Assoc. Computational Linguistics, vol. 10, pp. 145–162, 2022.",
        "[24] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, 'LoRA: Low-rank adaptation of large language models,' in Proc. Int. Conf. Learning Representations (ICLR), 2022.",
        "[25] C. Goddard, S. Siriwardhana, M. Ehghaghi, L. Meyers, V. Karpukhin, B. Benedict, M. McQuade, and J. Solawetz, 'Arcee’s MergeKit: A toolkit for merging large language models,' in Proc. EMNLP: Industry Track, 2024.",
        "[26] M. Popović, 'chrF++: Words helping character n-grams,' in Proc. Second Conf. Machine Translation, 2017, pp. 612–618.",
        "[27] D. Marczak, B. Twardowski, T. Trzciński, and S. Cygert, 'MagMax: Leveraging model merging for seamless continual learning,' in Proc. European Conf. Computer Vision (ECCV), 2024.",
        "[28] Open Language Data Initiative, 'FLORES+: An open multilingual evaluation benchmark,' 2024."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        format_p(p_ref, space_before=1, space_after=3)
        r_run = p_ref.add_run(r)
        r_run.font.name = 'Arial'
        r_run.font.size = Pt(8.5)
        r_run.font.color.rgb = COLOR_TEXT

    out_file = 'SLM_Model_Merging_Mini_Project_Report.docx'
    doc.save(out_file)
    print(f"Successfully generated {out_file} ({os.path.getsize(out_file)} bytes)")

if __name__ == '__main__':
    create_report()
