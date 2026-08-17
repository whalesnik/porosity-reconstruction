"""Почему наши поровые тела мельче и рванее, чем в статье?

Две гипотезы:
  A. Разный участок объёма (порода неоднородна) - тогда разброс между
     подкубами должен быть того же порядка, что и расхождение со статьёй.
  B. Мы режем порогом СЫРОЙ скан, а в статье перед Random Walker идёт
     цепочка bandpass + bilateral + non-local means. Порог по шумным данным
     дробит связные объекты на куски и раздувает площадь поверхности.

Обе проверяются количественно:
  - число связных компонент на единицу объёма (дробление),
  - удельная поверхность S/V (рваность границы): у шумной сегментации она
    завышена в разы при той же пористости,
  - доля объёма в наибольшем кластере (связность).

Пористость при этом ОДИНАКОВА по построению (порог подбирается под 6 %),
поэтому все различия - чисто морфологические.
"""
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.restoration import denoise_bilateral, denoise_nl_means

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AIR = "hgxdh8ps94-1/Air/Air_%04d.png"
SUB = 120
STRUCT = ndi.generate_binary_structure(3, 1)


def load_cube(z0, y0, x0, n=SUB):
    return np.stack([imread(AIR % z).astype(np.float32)[y0:y0 + n, x0:x0 + n]
                     for z in range(z0, z0 + n)])


def thr_for_fraction(vol, frac=6.0):
    """Порог, дающий заданную долю объёма - чтобы сравнивать морфологию
    при одинаковой пористости, а не пористость."""
    return float(np.percentile(vol, frac))


def morphology(mask):
    lab, n = ndi.label(mask, structure=STRUCT)
    if n == 0:
        return dict(n_comp=0, spec_surf=0.0, largest=0.0, median_size=0.0)
    sizes = np.bincount(lab.ravel())[1:]
    # удельная поверхность: число граней между порой и не-порой
    faces = 0
    for ax in range(3):
        faces += int((np.diff(mask.astype(np.int8), axis=ax) != 0).sum())
    return dict(n_comp=int(n),
                spec_surf=faces / max(mask.sum(), 1),
                largest=float(sizes.max() / mask.sum() * 100),
                median_size=float(np.median(sizes)))


# ===================== A: разброс между участками объёма =====================
print("=== A. разброс между разными подкубами (сырой порог, всегда 6 % объёма) ===")
print(f"{'позиция':>22} {'компонент':>10} {'S/V':>7} {'крупнейший':>11} {'медиана,вокс':>13}")
print("-" * 68)
positions = [(390, 390, 390), (150, 150, 150), (150, 600, 600),
             (650, 200, 650), (650, 650, 200)]
for (z0, y0, x0) in positions:
    v = load_cube(z0, y0, x0)
    m = v < thr_for_fraction(v)
    r = morphology(m)
    print(f"{str((z0,y0,x0)):>22} {r['n_comp']:>10d} {r['spec_surf']:>7.2f} "
          f"{r['largest']:>10.1f}% {r['median_size']:>13.0f}")

# ===================== B: влияние денойзинга ================================
print("\n=== B. тот же подкуб, разная предобработка перед порогом ===")
v = load_cube(390, 390, 390)
# устойчивая оценка шума без PyWavelets: MAD остатка после медианы 3x3x3
res = v - ndi.median_filter(v, size=3)
sig = float(np.median(np.abs(res - np.median(res))) / 0.6745 * 1.5)
print(f"оценка шума сухого скана: sigma = {sig:.0f} "
      f"(контраст кварц-пора всего 1789, то есть 7 sigma на весь диапазон)\n")

variants = {}
variants["сырой (то, что было)"] = v
variants["median 3x3x3"] = ndi.median_filter(v, size=3)
variants["gauss sigma=1"] = ndi.gaussian_filter(v, 1.0)

# bandpass как в статье: вычесть крупномасштабный фон
bp = v - ndi.gaussian_filter(v, 12.0) + float(v.mean())
variants["bandpass + median3"] = ndi.median_filter(bp, size=3)

# bilateral: сохраняет границы зёрен, давит шум внутри них
sc = (v - v.min()) / (v.max() - v.min())
bil = np.stack([denoise_bilateral(sc[k], sigma_color=sig / (v.max() - v.min()) * 1.6,
                                  sigma_spatial=2.0) for k in range(SUB)])
variants["bilateral (по срезам)"] = bil * (v.max() - v.min()) + v.min()

# non-local means в 3D - ближе всего к тому, что делали в статье
nlm = denoise_nl_means(sc, patch_size=3, patch_distance=4,
                       h=0.8 * sig / (v.max() - v.min()), fast_mode=True,
                       channel_axis=None)
variants["non-local means 3D"] = nlm * (v.max() - v.min()) + v.min()

print(f"{'вариант':>24} {'компонент':>10} {'S/V':>7} {'крупнейший':>11} {'медиана,вокс':>13}")
print("-" * 70)
base = None
for name, vol in variants.items():
    m = vol < thr_for_fraction(vol)
    r = morphology(m)
    if base is None:
        base = r
    print(f"{name:>24} {r['n_comp']:>10d} {r['spec_surf']:>7.2f} "
          f"{r['largest']:>10.1f}% {r['median_size']:>13.0f}")

print(f"\nS/V - удельная поверхность: сколько граней приходится на один поровый")
print(f"воксель. У идеального компактного тела она стремится к ~1-2, у рваной")
print(f"шумовой сегментации доходит до 4-6. Пористость во всех строках одна.")
