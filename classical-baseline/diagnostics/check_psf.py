"""Одинакова ли резкость сухого и Xe-скана?

Если у Xe-скана шире PSF, разность даёт ореолы вокруг пор, и субразрешённая
пористость окажется частично артефактом размытия, а не физикой. На
ИНТЕГРАЛЬНУЮ пористость это не влияет (свёртка сохраняет среднее), но
портит повоксельную phi-карту, связность и 3D-модель - то есть ровно то,
что мы отдаём дальше.

Меряем радиальный спектр мощности: размытие давит высокие частоты, и
отношение спектров двух сканов покажет это прямо.
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
ZS = [150, 300, 450, 600, 750]
N = 900

yy, xx = np.mgrid[:N, :N]
r = np.hypot(yy - N / 2, xx - N / 2).astype(int)
rbin = np.bincount(r.ravel())
win = np.outer(np.hanning(N), np.hanning(N))     # окно против утечки спектра

acc = {"air": [], "xe": []}
grad = {"air": [], "xe": []}
for z in ZS:
    for key, pat in (("air", AIR), ("xe", XE)):
        a = imread(pat % z).astype(np.float32)
        a = (a - a.mean()) * win
        p = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
        acc[key].append(np.bincount(r.ravel(), p.ravel()) / rbin)
        g = ndi.gaussian_gradient_magnitude(imread(pat % z).astype(np.float32), 1.0)
        grad[key].append(float(g.mean()))

pa = np.mean(acc["air"], axis=0)
px = np.mean(acc["xe"], axis=0)
freq = np.arange(len(pa)) / N          # циклов на воксель

print(f"средний модуль градиента: Air = {np.mean(grad['air']):.1f}, "
      f"Xe = {np.mean(grad['xe']):.1f}, отношение = {np.mean(grad['xe'])/np.mean(grad['air']):.3f}")

print("\nотношение спектров мощности Xe/Air по частотам:")
for f_target in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
    k = int(f_target * N)
    if k < len(pa):
        print(f"  f = {f_target:.2f} цикл/вокс (период {1/f_target:4.1f} вокс, "
              f"{3/f_target:5.1f} мкм): Xe/Air = {px[k]/pa[k]:6.3f}")

# оценка дополнительного гауссова размытия Xe относительно Air:
# отношение амплитуд = exp(-2 pi^2 sigma^2 f^2), берём наклон log(ratio) по f^2
m = (freq > 0.05) & (freq < 0.30) & (pa > 0) & (px > 0)
ratio = px[m] / pa[m]
coef = np.polyfit(freq[m] ** 2, np.log(np.maximum(ratio, 1e-12)), 1)
sigma_extra = np.sqrt(max(-coef[0], 0) / (2 * (2 * np.pi ** 2)))
print(f"\nэквивалентное доп. размытие Xe относительно Air: sigma ~ {sigma_extra:.2f} вокс "
      f"({sigma_extra*3:.1f} мкм)")
print("  (0 = сканы одинаково резкие; >0.5 вокс - разностная карта будет с ореолами)")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].loglog(freq[1:N//2], pa[1:N//2], label="Air")
ax[0].loglog(freq[1:N//2], px[1:N//2], label="Xe")
ax[0].set_xlabel("частота, цикл/воксель"); ax[0].set_ylabel("мощность")
ax[0].legend(); ax[0].grid(alpha=.3, which="both"); ax[0].set_title("радиальные спектры мощности")

ax[1].semilogx(freq[1:N//2], px[1:N//2] / pa[1:N//2], lw=1.4)
ax[1].axhline(1, c="k", ls="--", lw=1)
ax[1].set_xlabel("частота, цикл/воксель"); ax[1].set_ylabel("Xe / Air")
ax[1].set_ylim(0, 3); ax[1].grid(alpha=.3, which="both")
ax[1].set_title("отношение спектров\n<1 на высоких частотах = Xe размытее")

z = 450
a = imread(AIR % z).astype(np.float32)[400:500, 400:500]
x = imread(XE % z).astype(np.float32)[400:500, 400:500]
ax[2].plot(a[50], label="Air"); ax[2].plot(x[50], label="Xe")
ax[2].set_xlabel("x, воксель"); ax[2].set_ylabel("интенсивность")
ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_title("профиль одной строки (100 вокс)")
plt.tight_layout(); plt.savefig("results/diag_psf.png", dpi=95)
print("saved results/diag_psf.png")
