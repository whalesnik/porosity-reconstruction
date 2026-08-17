"""Визуализации референсного результата для созвона."""
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
Z = 450

cal = json.load(open("results/calibration.json"))
st = json.load(open("results/stats.json"))
sens = json.load(open("results/sensitivity.json"))
BETA, D100, THR = cal["beta"], cal["d100"], cal["thr_pore"]

air = imread(AIR % Z).astype(np.float32)
xe = imread(XE % Z).astype(np.float32)
dI = ndi.median_filter(xe - air, 3) + BETA
phi = dI / D100 * 100.0
pores = air < THR

# ---------------------------------------------------------------- рисунок 1
fig, ax = plt.subplots(2, 3, figsize=(17.5, 11.6))
vmin, vmax = np.percentile(air, [0.5, 99.5])

ax[0, 0].imshow(air, cmap="gray", vmin=vmin, vmax=vmax)
ax[0, 0].set_title(f"1. Сухой скан (Air), z={Z}\nзёрна светлые, поры тёмные")

ax[0, 1].imshow(xe, cmap="gray", vmin=vmin, vmax=vmax)
ax[0, 1].set_title("2. Скан с ксеноном (Xe)\nпоры подсвечены, стали ярче матрицы")

im = ax[0, 2].imshow(dI, cmap="inferno", vmin=0, vmax=D100)
ax[0, 2].set_title(f"3. Разность dI = Xe - Air + beta\nминералогия сократилась, "
                   f"остался только Xe-сигнал")
plt.colorbar(im, ax=ax[0, 2], fraction=.046, label="dI")

g = np.clip((air - vmin) / (vmax - vmin), 0, 1)
rgb = np.dstack([g, g, g])
rgb[pores] = [1, 0.2, 0.1]
ax[1, 0].imshow(rgb)
ax[1, 0].set_title(f"4. Разрешённые поры (классика uxCT)\n"
                   f"порог phi=50%: I<{THR:.0f} -> {st['resolved_porosity']:.2f} % объёма")

im = ax[1, 1].imshow(np.clip(phi, 0, 100), cmap="viridis", vmin=0, vmax=100)
ax[1, 1].set_title(f"5. phi-map: пористость каждого вокселя\n"
                   f"полная пористость {st['total_porosity']:.2f} %")
plt.colorbar(im, ax=ax[1, 1], fraction=.046, label="phi, %")

# что видит ТОЛЬКО Xe-метод: пористость там, где сухой скан показывает матрицу
hidden = (~pores) & (phi > 25)
rgb2 = np.dstack([g, g, g])
rgb2[hidden] = [0.1, 1.0, 0.3]
rgb2[pores] = [1, 0.2, 0.1]
ax[1, 2].imshow(rgb2)
ax[1, 2].set_title("6. Красное - видно и без ксенона.\nЗелёное - субразрешённая "
                   "пористость,\nв сухом скане неотличима от матрицы")

for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("Xe-enhanced micro-CT: классический референсный пайплайн (Ebadi et al., 2022)",
             fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig1_pipeline.png", dpi=95)
plt.close()

# ---------------------------------------------------------------- рисунок 2
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

cen = np.array(st["phi_hist_centers"]); h = np.array(st["phi_hist"], dtype=float)
ax[0].semilogy(cen, h, lw=1.4)
ax[0].axvline(0, c="k", lw=1)
ax[0].axvline(100, c="r", ls="--", lw=1.3, label="phi=100% (открытая пора)")
ax[0].axvspan(-60, 0, color="grey", alpha=.2, label="отрицательные = шум")
ax[0].set_xlim(-50, 140); ax[0].set_xlabel("phi, %"); ax[0].set_ylabel("вокселей")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[0].set_title("распределение пористости по вокселям\nхвост гаснет на 100% - "
                "калибровка dI100 верна")

T = [t for t, _ in st["truncation_curve"]]
V = [v for _, v in st["truncation_curve"]]
ax[1].plot(T, V, "o-", lw=2)
ax[1].axhline(13.6, c="m", ls="--", lw=1.4, label="лаборатория 13.6 %")
ax[1].axhline(11.1, c="r", ls="--", lw=1.4, label="Xe-пайплайн статьи 11.1 %")
ax[1].axvline(20, c="grey", ls=":", lw=1.2)
ax[1].set_xlabel("отбрасываем воксели с phi ниже, %")
ax[1].set_ylabel("интегральная пористость, %")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
ax[1].set_title("почему в статье 11.1 %, а в лаборатории 13.6 %\n"
                "порог уверенного сигнала срезает диффузный хвост")

zs = np.array(st["zs"]); pz = np.array(st["phi_per_slice"])
rz = np.array(st["resolved_per_slice"])
ax[2].plot(zs, pz, lw=1.2, label="полная phi")
ax[2].plot(zs, rz, lw=1.2, label="разрешённая")
ax[2].fill_between(zs, rz, pz, alpha=.25, label="субразрешённая")
ax[2].axhline(13.6, c="m", ls="--", lw=1.2)
ax[2].set_xlabel("z, срез"); ax[2].set_ylabel("пористость, %")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)
ax[2].set_title(f"пористость по высоте образца\nразброс {pz.std():.2f} п.п. - "
                f"полосатости по z нет")
plt.tight_layout(); plt.savefig("results/fig2_analysis.png", dpi=95)
plt.close()

print(f"phi = {sens['phi']:.2f} +- {sens['err_total']:.2f} %")
print("saved results/fig1_pipeline.png, results/fig2_analysis.png")
