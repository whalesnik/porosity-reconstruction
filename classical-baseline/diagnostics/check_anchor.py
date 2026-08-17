"""Мода или медиана для опоры "кварц"? Расхождение стоит 1.07 п.п. пористости.

Теория: чистый кварц даёт симметричный шумовой пик на dI = -beta. Если в
окно опоры попадает микропористая фаза, она добавляет ОДНОСТОРОННИЙ хвост
справа (у неё dI > 0). Тогда медиана смещена, мода - нет.

Проверяем это прямо: смотрим форму распределения dI по опорным вокселям и
сравниваем три оценки центра. Плюс независимый контроль - опора на плотных
включениях: при верной beta обе твёрдые опоры обязаны дать dI = 0, поэтому
правильная оценка должна СБЛИЖАТЬ их, а не разводить.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.io import imread

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AIR = "hgxdh8ps94-1/Air/Air_%04d.png"
XE = "hgxdh8ps94-1/Xenon_/Xe_%04d.png"
ZS = np.linspace(6, 893, 16).astype(int)
I_QUARTZ = 8859.5
SE3 = np.ones((3, 3, 3), bool)

LO, HI, NB = -2000.0, 2000.0, 500
hq = np.zeros(NB, dtype=np.int64)
hd = np.zeros(NB, dtype=np.int64)

for z in ZS:
    a = np.stack([imread(AIR % zz).astype(np.float32) for zz in (z - 1, z, z + 1)])
    x = np.stack([imread(XE % zz).astype(np.float32) for zz in (z - 1, z, z + 1)])
    dI = ndi.median_filter(x[1] - a[1], 3)
    for mask, h in ((np.abs(a - I_QUARTZ) < 200, hq), (a > 12000, hd)):
        m = ndi.binary_erosion(mask, SE3)[1]
        if m.sum() < 50:
            continue
        idx = np.clip((dI[m] - LO) / (HI - LO) * NB, 0, NB - 1).astype(np.int64)
        h += np.bincount(idx, minlength=NB)

cen = np.linspace(LO, HI, NB + 1)[:-1] + (HI - LO) / NB / 2


def stats(h):
    n = h.sum()
    c = np.cumsum(h) / n
    med = float(np.interp(0.5, c, cen))
    mean = float((h * cen).sum() / n)
    i = int(h.argmax())
    y0, y1, y2 = float(h[i-1]), float(h[i]), float(h[i+1])
    d = y0 - 2 * y1 + y2
    mode = float(cen[i] + (0.5 * (y0 - y2) / d if d else 0) * (cen[1] - cen[0]))
    # асимметрия: сравниваем массу справа и слева от моды
    left = float(h[cen < mode].sum()); right = float(h[cen > mode].sum())
    return dict(n=int(n), mode=mode, median=med, mean=mean,
                skew_ratio=right / max(left, 1))


sq, sd = stats(hq), stats(hd)
print("=== опора КВАРЦ (окно +-200 вокруг пика, 3D-эрозия) ===")
print(f"  вокселей: {sq['n']:,}")
print(f"  мода    = {sq['mode']:+8.1f}")
print(f"  медиана = {sq['median']:+8.1f}   (сдвиг от моды {sq['median']-sq['mode']:+.1f})")
print(f"  среднее = {sq['mean']:+8.1f}   (сдвиг от моды {sq['mean']-sq['mode']:+.1f})")
print(f"  масса справа/слева от моды = {sq['skew_ratio']:.3f}")
print(f"  -> хвост {'СПРАВА' if sq['skew_ratio']>1.05 else 'симметрично'}: "
      f"{'есть загрязнение пористой фазой, медиана смещена' if sq['skew_ratio']>1.05 else 'обе оценки годятся'}")

print("\n=== опора ПЛОТНЫЕ ВКЛЮЧЕНИЯ (I_air > 12000) ===")
print(f"  вокселей: {sd['n']:,}")
print(f"  мода    = {sd['mode']:+8.1f}")
print(f"  медиана = {sd['median']:+8.1f}")

print("\n=== согласие двух ТВЁРДЫХ опор при разных оценках beta ===")
print("  (обе фазы непористы, значит после нормализации обе обязаны дать dI = 0)")
print(f"{'оценка':>12} {'beta':>9} {'кварц':>9} {'плотные':>10} {'|расхождение|':>15}")
print("-" * 60)
for name, key in (("мода", "mode"), ("медиана", "median")):
    beta = -sq[key]
    q_res = sq[key] + beta
    d_res = sd[key] + beta
    print(f"{name:>12} {beta:>9.1f} {q_res:>9.1f} {d_res:>10.1f} {abs(d_res-q_res):>15.1f}")

print("\nЧья оценка ближе к истине - решает вторая опора: правильная beta")
print("должна обнулить ОБЕ, а не только ту, по которой её считали.")

plt.figure(figsize=(12, 4.6))
plt.subplot(1, 2, 1)
plt.plot(cen, hq, lw=1.4)
plt.axvline(sq["mode"], c="C2", lw=1.6, label=f"мода {sq['mode']:.0f}")
plt.axvline(sq["median"], c="C3", lw=1.6, ls="--", label=f"медиана {sq['median']:.0f}")
plt.xlim(-900, 900); plt.legend(); plt.grid(alpha=.3)
plt.xlabel("dI (без нормализации)"); plt.ylabel("вокселей")
plt.title("опора кварц: форма распределения")
plt.subplot(1, 2, 2)
plt.semilogy(cen, hq, lw=1.4)
plt.axvline(sq["mode"], c="C2", lw=1.6); plt.axvline(sq["median"], c="C3", lw=1.6, ls="--")
plt.xlim(-1200, 2000); plt.grid(alpha=.3, which="both")
plt.xlabel("dI"); plt.title("то же в лог-масштабе: виден правый хвост")
plt.tight_layout(); plt.savefig("results/diag_anchor.png", dpi=95)
print("\nsaved results/diag_anchor.png")
