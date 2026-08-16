"""
Эксперимент 0 — посмотреть на данные глазами, прежде чем им доверять.

Повод: в exp01 маска керна по Отсу дала долю "внутри керна" = 0.00, чего
физически быть не может. Значит, либо фон не тёмный, либо у изображения
другая геометрия, чем предполагалось. Прежде чем чинить код — смотрим на срезы
и гистограммы.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib_dino as L

OUT = L.RUNS_DIR / "exp00_inspect"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    air, air_idx = L.load_air(6)
    xe, xe_idx = L.load_xe(6)
    print("Air:", air.shape, air.dtype, "Xe:", xe.shape)

    fig, axes = plt.subplots(3, 6, figsize=(19, 9.5))
    for j in range(6):
        a, x = air[j], xe[j]
        axes[0, j].imshow(a, cmap="gray")
        axes[0, j].set_title(f"Air #{air_idx[j]}\n[{a.min():.0f},{a.max():.0f}] mean={a.mean():.0f}", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(x, cmap="gray")
        axes[1, j].set_title(f"Xe #{xe_idx[j]}\n[{x.min():.0f},{x.max():.0f}] mean={x.mean():.0f}", fontsize=8)
        axes[1, j].axis("off")
        axes[2, j].hist(a.ravel(), bins=120, alpha=0.6, label="air", log=True)
        axes[2, j].hist(x.ravel(), bins=120, alpha=0.6, label="xe", log=True)
        axes[2, j].legend(fontsize=7)
        axes[2, j].tick_params(labelsize=6)
    fig.suptitle("Сырые срезы Air / Xe и гистограммы интенсивностей")
    fig.tight_layout()
    fig.savefig(OUT / "raw_slices.png", dpi=130)

    # Диагностика порога Отсу на среднем срезе
    from skimage.filters import threshold_otsu
    mid = air[len(air) // 2]
    thr = threshold_otsu(mid)
    above = mid > thr
    print(f"Otsu порог={thr:.0f}; доля пикселей выше порога={above.mean():.4f}")
    print(f"Перцентили среза: {np.percentile(mid, [1, 5, 25, 50, 75, 95, 99]).round(0)}")

    # Профиль яркости по горизонтали через центр — видно, есть ли фон по краям
    row = mid[mid.shape[0] // 2]
    fig2, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].imshow(mid, cmap="gray")
    ax[0].axhline(mid.shape[0] // 2, c="r", lw=0.8)
    ax[0].set_title("средний срез Air")
    ax[1].plot(row, lw=0.7)
    ax[1].axhline(thr, c="r", ls="--", label=f"Otsu={thr:.0f}")
    ax[1].set_title("профиль яркости по красной линии")
    ax[1].legend(fontsize=8)
    ax[2].imshow(above, cmap="gray")
    ax[2].set_title(f"маска > Otsu (доля={above.mean():.3f})")
    fig2.tight_layout()
    fig2.savefig(OUT / "otsu_diagnostic.png", dpi=130)

    # Разностная карта Xe - Air: то, ради чего весь проект
    d = xe[len(xe) // 2].astype(np.float32) - air[len(air) // 2].astype(np.float32)
    fig3, ax3 = plt.subplots(1, 3, figsize=(16, 5))
    ax3[0].imshow(air[len(air) // 2], cmap="gray"); ax3[0].set_title("Air"); ax3[0].axis("off")
    ax3[1].imshow(xe[len(xe) // 2], cmap="gray"); ax3[1].set_title("Xe"); ax3[1].axis("off")
    im = ax3[2].imshow(d, cmap="coolwarm", vmin=-np.abs(d).max() * 0.5, vmax=np.abs(d).max() * 0.5)
    ax3[2].set_title(f"Xe − Air (mean={d.mean():+.0f})"); ax3[2].axis("off")
    fig3.colorbar(im, ax=ax3[2])
    fig3.tight_layout()
    fig3.savefig(OUT / "difference_map.png", dpi=130)
    print(f"Разность Xe-Air на среднем срезе: mean={d.mean():+.1f}, std={d.std():.1f}")
    print(f"Готово -> {OUT}")


if __name__ == "__main__":
    main()
