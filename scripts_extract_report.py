import fitz
import os

pdf_path = r'C:\Users\12403\.gemini\antigravity-ide\brain\70d9d7a4-7540-470c-8a5c-0ae5508d9792\.user_uploaded\media_1792556988860.pdf'
doc = fitz.open(pdf_path)
print(f'Total pages: {len(doc)}')

os.makedirs('report_assets', exist_ok=True)
img_count = 0
for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images(full=True)
    for img_idx, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image['image']
        image_ext = base_image['ext']
        w = base_image['width']
        h = base_image['height']
        img_name = f'page_{page_num+1}_img_{img_idx+1}.{image_ext}'
        img_path = os.path.join('report_assets', img_name)
        with open(img_path, 'wb') as f:
            f.write(image_bytes)
        img_count += 1
        print(f'Page {page_num+1}: {img_name} ({w}x{h})')

print(f'Total extracted images: {img_count}')
