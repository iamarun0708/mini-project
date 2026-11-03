import docx

def audit_doc(path, label):
    doc = docx.Document(path)
    print(f"\n=== {label} ({path}) ===")
    s = doc.sections[0]
    print(f"  Page: {s.page_width.inches:.3f} x {s.page_height.inches:.3f} in")
    print(f"  Margins: L={s.left_margin.inches:.3f} in (35mm), R={s.right_margin.inches:.3f} in (20mm), T={s.top_margin.inches:.3f} in (30mm), B={s.bottom_margin.inches:.3f} in (25mm)")
    print(f"  Paragraphs: {len(doc.paragraphs)}, Tables: {len(doc.tables)}")

    images = 0
    for p in doc.paragraphs:
        if 'blip' in p._p.xml or 'graphicData' in p._p.xml:
            images += 1
    print(f"  Embedded Images: {images}")

audit_doc('reports/Batch -14 Front Pages.docx', 'Front Pages Document')
audit_doc('reports/Batch -14 Content.docx', 'Content Report Document')
audit_doc('reports/Batch -14 Report.docx', 'Complete Master Report')
