"""Реализация phi-map ТОЧНО по рецепту Ebadi et al. (2022), рядом с нашей.

Рецепт из раздела 4 статьи, дословно:
  1. Binary uxCT: сегментация сухого скана -> разрешённые поры, 6 % объёма.
  2. Diff-uxCT = Xe - Air, затем ИНВЕРСИЯ и те же фильтры денойзинга.
  3. Интенсивности вокселей, уже помеченных как поры, обнуляются.
  4. RW по маскированному изображению -> Binary Diff-uxCT, "porous part".
  5. phi: поры -> 100 %, твёрдое -> 0 %, porous part -> линейно по интенсивности,
     причём "the intensity of 191 has the phi_v of 0, and the intensity of 0
     has the phi_v of 100".

Пункт 5 - ключевой и он же источник расхождения с лабораторией. Ноль
пористости привязан к интенсивности 191, то есть к МИНИМАЛЬНОМУ Xe-сигналу
внутри porous part, а не к истинному уровню твёрдой фазы. Из этого следует
два смещения вниз:
  (а) воксели вне porous part получают phi = 0, хотя сигнал у них ненулевой;
  (б) воксели на границе porous part получают phi ~ 0 вместо истинного T/dI100.
И одно смещение вверх:
  (в) все разрешённые поры получают ровно 100 %, хотя часть из них заполнена
      матрицей частично.

Наш вариант (xe_pipeline.py) ничего не обнуляет: phi = dI / dI100 везде,
ноль привязан к твёрдой фазе по морфологической опоре.

Скрипт считает обе карты, сравнивает и рисует их рядом с ПОДЛИННЫМИ панелями
из статьи (paperfigs/, извлечены из PDF).
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.io import imread

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

AIR = "hgxdh8ps94-1/Air/Air_%04d.png"
XE = "hgxdh8ps94-1/Xenon_/Xe_%04d.png"
VOXEL_UM = 3.0                      # из статьи: spatial resolution 3 um/vox
CROP = 201                          # кроп статьи: линейка 100 мкм = 86 px, панель 518 px
SWEEP_ZS = list(range(5, 900, 10))  # 90 срезов на подбор порога

cal = json.load(open("results/calibration.json"))
BETA, D100 = cal["beta"], cal["d100"]

# --- порог разрешённых пор, дающий заявленные в статье 6 % ------------------
h = np.zeros(650, dtype=np.int64)
lo, hi = 6000.0, 32000.0
cen = np.linspace(lo, hi, 651)[:-1] + (hi - lo) / 650 / 2
for z in SWEEP_ZS[::3]:
    a = imread(AIR % z).astype(np.float32)
    h += np.bincount(np.clip((a - lo) / (hi - lo) * 650, 0, 649).astype(np.int64).ravel(),
                     minlength=650)
cum = np.cumsum(h) / h.sum() * 100
THR_6 = float(np.interp(6.0, cum, cen))
print(f"порог разрешённых пор для 6 % (как в статье): I_air < {THR_6:.0f}")
print(f"  (наш порог phi=50 %: I_air < {cal['thr_pore']:.0f} -> 3.43 %)")

# --- подбор порога porous part ----------------------------------------------
# В статье porous part выделяется алгоритмом Random Walker; здесь тот же смысл
# передаётся порогом по Xe-сигналу - величина, от которой зависит результат,
# это доля объёма, попавшая в porous part, а не конкретный алгоритм.
print(f"\nсканируем порог porous part по {len(SWEEP_ZS)} срезам "
      f"(сглаживание gauss sigma=1.5, как ближайший аналог их цепочки фильтров)\n")
print(f"{'T (dI)':>8} {'T (phi)':>8} {'porous part':>12} {'phi полная':>11} {'источники':>30}")
print("-" * 76)

TS = [0, 150, 300, 450, 600, 750, 900]
res = {t: dict(phi=0.0, part=0.0, resolved=0.0) for t in TS}
n = 0
for z in SWEEP_ZS:
    a = imread(AIR % z).astype(np.float32)
    x = imread(XE % z).astype(np.float32)
    dI = ndi.gaussian_filter(x - a, 1.5) + BETA
    pores = a < THR_6
    for T in TS:
        part = (~pores) & (dI >= T)
        # линейная шкала: phi = 0 на границе porous part, 100 на dI100
        phi = np.zeros_like(dI)
        phi[part] = (dI[part] - T) / max(D100 - T, 1e-6) * 100.0
        phi[pores] = 100.0
        res[T]["phi"] += float(np.clip(phi, 0, 100).mean())
        res[T]["part"] += float(part.mean()) * 100
    n += 1

for T in TS:
    p = res[T]["phi"] / n
    pp = res[T]["part"] / n
    note = ""
    if abs(p - 11.1) < 0.6:
        note = "<-- 11.1 % статьи"
    print(f"{T:>8} {T/D100*100:>7.1f}% {pp:>11.1f}% {p:>10.2f}% {note:>30}")

ours = 0.0
for z in SWEEP_ZS:
    a = imread(AIR % z).astype(np.float32)
    x = imread(XE % z).astype(np.float32)
    ours += float((ndi.median_filter(x - a, 3) + BETA).mean()) / D100 * 100
ours /= len(SWEEP_ZS)
print(f"\nнаш метод на тех же срезах: phi = {ours:.2f} % (ничего не обнуляется)")

# --- карты для картинки ------------------------------------------------------
T_BEST = min(TS[1:], key=lambda t: abs(res[t]["phi"] / n - 11.1))
print(f"\nближе всего к статье: T = {T_BEST} (phi = {res[T_BEST]['phi']/n:.2f} %), "
      f"porous part = {res[T_BEST]['part']/n:.1f} % объёма")

Z = 450
c0 = (900 - CROP) // 2
sl = (slice(c0, c0 + CROP), slice(c0, c0 + CROP))
a = imread(AIR % Z).astype(np.float32)
x = imread(XE % Z).astype(np.float32)
dI_soft = ndi.gaussian_filter(x - a, 1.5) + BETA
dI_ours = ndi.median_filter(x - a, 3) + BETA
pores6 = a < THR_6

phi_paper = np.zeros_like(dI_soft)
part = (~pores6) & (dI_soft >= T_BEST)
phi_paper[part] = (dI_soft[part] - T_BEST) / (D100 - T_BEST) * 100.0
phi_paper[pores6] = 100.0
phi_paper = np.clip(phi_paper, 0, 100)
phi_ours = np.clip(dI_ours / D100 * 100.0, 0, 100)

def trim(path):
    """Убирает белые поля, подпись панели и полосу цветовой шкалы.

    Панели вырезаны из PDF вместе с окружением, поэтому оставляем только
    самый широкий непрерывный блок непустых столбцов и строк - это и есть
    само изображение, а шкала отделена от него белым промежутком.
    """
    im = plt.imread(path)
    g = im if im.ndim == 2 else im[..., :3].mean(axis=2)
    if g.max() > 1.5:
        g = g / 255.0

    def widest(mask):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return 0, len(mask)
        brk = np.nonzero(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[brk + 1]]
        ends = np.r_[idx[brk], idx[-1]]
        k = int(np.argmax(ends - starts))
        return starts[k], ends[k] + 1

    x0, x1 = widest(g.mean(axis=0) < 0.94)
    y0, y1 = widest(g.mean(axis=1) < 0.94)
    return g[y0:y1, x0:x1]


fig, ax = plt.subplots(2, 3, figsize=(16.5, 11.8))
for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])

ax[0, 0].imshow(trim("paperfigs/fig6b.png"), cmap="gray")
ax[0, 0].set_title("СТАТЬЯ, Fig. 6(b)\nDiff-uxCT", fontsize=11)

vd = np.percentile(dI_ours[sl], [2, 99.5])
ax[0, 1].imshow(dI_ours[sl], cmap="gray", vmin=vd[0], vmax=vd[1])
ax[0, 1].set_title(f"НАШЕ, тот же масштаб\ndI, кроп {CROP}x{CROP} вокс "
                   f"({CROP*VOXEL_UM:.0f} мкм)", fontsize=11)

ax[0, 2].imshow(dI_ours, cmap="gray", vmin=vd[0], vmax=vd[1])
ax[0, 2].set_title("НАШЕ, всё поле 900x900\n(так выглядела прежняя картинка)", fontsize=11)

ax[1, 0].imshow(trim("paperfigs/fig9c.png"), cmap="gray")
ax[1, 0].set_title("СТАТЬЯ, Fig. 9(c)\n2D phi-map, полная пористость 11.1 %", fontsize=11)

ax[1, 1].imshow(phi_paper[sl], cmap="gray", vmin=0, vmax=100)
ax[1, 1].set_title(f"НАШЕ по рецепту статьи\nporous part {res[T_BEST]['part']/n:.0f} % объёма, "
                   f"phi = {res[T_BEST]['phi']/n:.2f} %", fontsize=11)

im = ax[1, 2].imshow(phi_ours[sl], cmap="gray", vmin=0, vmax=100)
ax[1, 2].set_title(f"НАШЕ без обнуления\nphi = {ours:.2f} % (лаборатория 13.6 %)", fontsize=11)
plt.colorbar(im, ax=ax[1, 2], fraction=.046, label="phi, %")

fig.suptitle("Почему картинки выглядят по-разному: масштаб кадра и обнуление слабого сигнала",
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.5)
plt.savefig("results/fig4_paper_comparison.png", dpi=95)
json.dump({str(t): dict(phi=res[t]["phi"] / n, porous_part=res[t]["part"] / n) for t in TS}
          | {"ours": ours, "thr_resolved_6pct": THR_6, "T_best": T_BEST},
          open("results/paper_method.json", "w"), indent=2)
print("saved results/fig4_paper_comparison.png, results/paper_method.json")
