# Панели из статьи

`paper_method.py` и `render_3d.py` строят сравнение с рисунками из
Ebadi et al., *Lift the veil of secrecy in sub-resolved pores by Xe-enhanced
computed tomography*, Fuel 328 (2022) 125274.

Сами панели сюда **не выложены**: статья закрытая, «© 2022 Elsevier Ltd.
All rights reserved». Извлеките их из своей легальной копии PDF.

Нужны четыре файла: `fig6b.png` (Diff-µxCT), `fig9a.png` (triple media),
`fig9c.png` (2D φ-map), `fig9d.png` (3D φ-map).

```python
import fitz, io
from PIL import Image

doc = fitz.open("Orlov CT.pdf")     # опубликованная версия в Fuel
# Fig. 6 - одна вклеенная картинка на стр. 7, две панели рядом
info = doc.extract_image(doc[6].get_images(full=True)[0][0])
fig6 = Image.open(io.BytesIO(info["image"])).convert("RGB")
fig6.crop((993, 0, 1511, 518)).save("paperfigs/fig6b.png")

# Fig. 9 - одна вклеенная картинка на стр. 10, четыре панели
info = doc.extract_image(doc[9].get_images(full=True)[0][0])
fig9 = Image.open(io.BytesIO(info["image"])).convert("RGB")
W, H = fig9.size
fig9.crop((int(.18*W), 0, int(.85*W), int(.37*H))).save("paperfigs/fig9a.png")
fig9.crop((1027, 884, 1622, 1394)).save("paperfigs/fig9c.png")
fig9.crop((int(.15*W), int(.60*H), int(.80*W), int(.99*H))).save("paperfigs/fig9d.png")
```

Координаты подобраны под конкретную вёрстку; если издание другое, проверьте
глазами. Скрипты обрезают белые поля и подписи сами.
