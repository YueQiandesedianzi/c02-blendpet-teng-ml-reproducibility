"""Render all Word-exported PDF pages and record their text."""
from pathlib import Path
import json
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
records=[]
for kind in ['ms','si']:
    directory=ROOT/'qa'/f'render-{kind}';doc=pdfium.PdfDocument(directory/f'{kind}.pdf');texts=[];thumbs=[]
    for i in range(len(doc)):
        page=doc[i];image=page.render(scale=1.55).to_pil();image.save(directory/f'page-{i+1:02d}.png')
        text=page.get_textpage().get_text_range();texts.append(f'\n--- PAGE {i+1} ---\n{text}')
        thumb=image.copy();thumb.thumbnail((360,510));thumbs.append(thumb)
        records.append(dict(document=kind,page=i+1,width=image.width,height=image.height,text_length=len(text),bottom_text=text[-200:]))
    (directory/'pages.txt').write_text(''.join(texts),encoding='utf-8')
    for start in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1080,1080),'#e6e9ec');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(thumbs[start:start+6]):
            x=(j%3)*360;y=(j//3)*540;sheet.paste(im,(x,y+22));draw.text((x+12,y+5),f'{kind.upper()} page {start+j+1}',fill='black')
        sheet.save(directory/f'contact-{start//6+1:02d}.png')
(ROOT/'qa/page-inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(pages=len(records),renderer='Microsoft Word PDF + pypdfium2',original_renderer_error='LibreOffice soffice.exe was not found on PATH')))
