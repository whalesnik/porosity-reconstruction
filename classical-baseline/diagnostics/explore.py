"""Разведка данных перед пайплайном.

Проверяем три вещи:
  1. Геометрия: где керн, где фон/держатель (керн круглый — фон в патчи брать нельзя).
  2. Гистограммы: видны ли горбы "поры / матрица / плотные включения" для seed-порогов RW.
  3. Смещение Air<->Xe по z: совпадают ли одноимённые срезы двух сканов физически.
     Если нет — 2D-обработка без 3D-регистрации даст мусор в разностной карте.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.io import imread

AIR = "hgxdh8ps94-1/Air/Air_%04d.png"
XE = "hgxdh8ps94-1/Xenon_/Xe_%04d.png"
N_Z = 900

# --- 1. геометрия и гистограммы на нескольких z -----------------------------
zs = [150, 450, 750]
fig, ax = plt.subplots(3, len(zs), figsize=(5 * len(zs), 14))
for j, z in enumerate(zs):
    air = imread(AIR % z).astype(np.float32)
    xe = imread(XE % z).astype(np.float32)
    ax[0, j].imshow(air, cmap="gray")
    ax[0, j].set_title(f"Air z={z}  [{air.min():.0f}..{air.max():.0f}]")
    ax[0, j].add_patch(plt.Rectangle((322, 322), 256, 256, ec="r", fc="none", lw=1.5))
    ax[1, j].imshow(xe - air, cmap="bwr", vmin=-1500, vmax=1500)
    ax[1, j].set_title(f"Xe - Air (без регистрации), z={z}")
    c = 128
    ac = air[450 - c:450 + c, 450 - c:450 + c]
    xc = xe[450 - c:450 + c, 450 - c:450 + c]
    ax[2, j].hist(ac.ravel(), bins=200, alpha=.6, label="Air")
    ax[2, j].hist(xc.ravel(), bins=200, alpha=.6, label="Xe")
    ax[2, j].set_yscale("log"); ax[2, j].legend(); ax[2, j].set_title(f"гистограмма центр.256, z={z}")
    print(f"z={z}: Air mean={air.mean():.0f} Xe mean={xe.mean():.0f} "
          f"| crop256 Air={ac.mean():.0f} Xe={xc.mean():.0f} diff={float((xc-ac).mean()):+.0f}")
    print(f"   Air crop perc {{p: v}}: "
          f"{ {p: round(float(np.percentile(ac, p)),0) for p in [1,5,15,25,50,75,95,99]} }")
plt.tight_layout(); plt.savefig("results/explore_slices.png", dpi=85); plt.close()

# --- 2. радиальный профиль: радиус керна ------------------------------------
air = imread(AIR % 450).astype(np.float32)
yy, xx = np.mgrid[:900, :900]
r = np.hypot(yy - 449.5, xx - 449.5).astype(int)
radial = np.bincount(r.ravel(), air.ravel()) / np.bincount(r.ravel())
plt.figure(figsize=(9, 4))
plt.plot(radial); plt.grid(alpha=.3)
plt.xlabel("радиус, px"); plt.ylabel("средняя интенсивность")
plt.title("радиальный профиль Air z=450 (край керна = обрыв)")
plt.savefig("results/explore_radial.png", dpi=90); plt.close()
drop = np.where(radial < radial[:200].mean() * 0.6)[0]
print("радиус, где интенсивность падает <60% от центральной:", drop[0] if len(drop) else "не найден")

# --- 3. z-смещение между сканами --------------------------------------------
# профиль среднего по срезу вдоль z (каждый 5-й срез) - если кривые сдвинуты,
# значит сканы смещены по z и одноимённые срезы НЕ соответствуют друг другу.
step = 5
zi = np.arange(0, N_Z, step)
prof_air = np.array([imread(AIR % z)[300:600, 300:600].mean() for z in zi])
prof_xe = np.array([imread(XE % z)[300:600, 300:600].mean() for z in zi])
pa = (prof_air - prof_air.mean()) / prof_air.std()
px = (prof_xe - prof_xe.mean()) / prof_xe.std()
xc = np.correlate(px, pa, mode="full")
lag = (xc.argmax() - (len(pa) - 1)) * step
print(f"оценка сдвига по z (кросс-корреляция z-профилей): {lag:+d} срезов")

plt.figure(figsize=(11, 4))
plt.plot(zi, prof_air, label="Air")
plt.plot(zi, prof_xe, label="Xe")
plt.xlabel("z, срез"); plt.ylabel("средняя интенсивность центр.300x300")
plt.legend(); plt.grid(alpha=.3); plt.title(f"z-профили двух сканов (оценка сдвига {lag:+d} срезов)")
plt.savefig("results/explore_zprofile.png", dpi=90); plt.close()
print("saved: results/explore_slices.png, explore_radial.png, explore_zprofile.png")
