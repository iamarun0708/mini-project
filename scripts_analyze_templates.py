import re

def analyze_rtf(path, name):
    print(f"\n==================== {name} ====================")
    with open(path, 'rb') as f:
        raw = f.read()

    # extract margins
    margl = re.search(rb'\\margl(\d+)', raw)
    margr = re.search(rb'\\margr(\d+)', raw)
    margt = re.search(rb'\\margt(\d+)', raw)
    margb = re.search(rb'\\margb(\d+)', raw)
    paperw = re.search(rb'\\paperw(\d+)', raw)
    paperh = re.search(rb'\\paperh(\d+)', raw)

    print("Page Setup:")
    if paperw and paperh:
        print(f"  Paper: {int(paperw.group(1))/1440:.2f} in x {int(paperh.group(1))/1440:.2f} in (w={paperw.group(1).decode()}, h={paperh.group(1).decode()})")
    if margl and margr and margt and margb:
        print(f"  Margins: Left={int(margl.group(1))/1440:.2f} in ({int(margl.group(1))/56.7:.1f} mm), Right={int(margr.group(1))/1440:.2f} in ({int(margr.group(1))/56.7:.1f} mm), Top={int(margt.group(1))/1440:.2f} in ({int(margt.group(1))/56.7:.1f} mm), Bottom={int(margb.group(1))/1440:.2f} in ({int(margb.group(1))/56.7:.1f} mm)")

    # clean picts
    clean = re.sub(rb'\{\\\*?\\pict[\s\S]*?\}', b'', raw)
    clean = re.sub(rb'\\pict[\s\S]*?\\par', b'', clean)

    # find pages
    pages = clean.split(b'\\page')
    print(f"Total pages: {len(pages)}")

    for i, page in enumerate(pages):
        # find text in this page
        # strip rtf commands
        p_text = re.sub(rb'\\[a-zA-Z0-9\-]+ ?', b' ', page)
        p_text = re.sub(rb'[{}]', b'', p_text)
        lines = [l.strip().decode('latin-1', errors='ignore') for l in p_text.splitlines() if l.strip()]
        # filter out empty or hex
        clean_lines = [l for l in lines if len(l) > 2 and not re.match(r'^[0-9a-fA-F\s]{20,}$', l)]
        # keep only meaningful text
        meaningful = []
        for l in clean_lines:
            # check if line has letters
            if re.search(r'[a-zA-Z]', l):
                meaningful.append(l)
        
        print(f"\n--- Page {i+1} ({len(meaningful)} meaningful lines) ---")
        for l in meaningful[:15]:
            print("  ", l[:100])

analyze_rtf('reports/Front Pages - Final template.rtf.doc', 'Front Pages - Final template.rtf.doc')
