"""
Эксперимент 6 (часть 1): визуальный осмотр СВОЕГО датасета до всякого моделирования.

Это первый раз, когда в проекте появились данные не из статьи Сколтеха, а свои
(образец 22322-24, сухой скан, воксель 1.84 мкм, 2775x2774x2420).
Прежде чем гонять detector, надо глазами убедиться в базовых вещах:
  - есть ли фон вокруг керна (у Сколтеха его не было — объём обрезан внутри образца);
  - совпадает ли физический масштаб (мкм на воксель) — если нет, сравнение
    текстур некорректно без пересэмплирования;
  - как выглядят гистограммы двух датасетов рядом.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib_dino as L

OWN_DIR = L.PROJECT_ROOT / "22322-24_dry_4mm_1.8um_16bit"
OUT = L.RUNS_DIR / "exp06_own_dataset"


def load_own(n: int | None = None) -> tuple[np.ndarray, list[str]]:
    files = sorted(OWN_DIR.glob("rec_*.tif"), key=lambda p: int(p.stem.split("_")[-1]))
    if n:
        idx = np.linspace(0, len(files) - 1, n, dtype=int)
        files = [files[i] for i in idx]
    vols = np.stack([np.array(Image.open(f)).astype(np.float32) for f in files])
    return vols, [f.name for f in files]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    own, names = load_own()
    sk, _ = L.load_air(12)
    print(f"свой:     {own.shape}  min={own.min():.0f} max={own.max():.0f} mean={own.mean():.0f}")
    print(f"Сколтех:  {sk.shape}  min={sk.min():.0f} max={sk.max():.0f} mean={sk.mean():.0f}")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, i in zip(axes[0], [0, len(own) // 3, 2 * len(own) // 3, len(own) - 1]):
        ax.imshow(own[i], cmap="gray")
        ax.set_title(f"свой: {names[i]}", fontsize=9)
        ax.axis("off")
    for ax, i in zip(axes[1], [0, 3, 7, 11]):
        ax.imshow(sk[i], cmap="gray")
        ax.set_title(f"Сколтех Air {i}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Свой датасет (сверху) против Сколтеха (снизу) — целые срезы")
    fig.tight_layout()
    fig.savefig(OUT / "slices_side_by_side.png", dpi=120, bbox_inches="tight")

    # центральные кропы одинакового ПИКСЕЛЬНОГО размера — так видно разницу текстуры
    fig2, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, i in zip(axes[0], [0, len(own) // 3, 2 * len(own) // 3, len(own) - 1]):
        r, c, s = L.center_box(own[i].shape, 0.12)
        ax.imshow(own[i][r:r + 224, c:c + 224], cmap="gray")
        ax.set_title(f"свой 224x224 (1.84 мкм/вокс)", fontsize=9)
        ax.axis("off")
    for ax, i in zip(axes[1], [0, 3, 7, 11]):
        r, c, s = L.center_box(sk[i].shape, 0.3)
        ax.imshow(sk[i][r:r + 224, c:c + 224], cmap="gray")
        ax.set_title("Сколтех 224x224", fontsize=9)
        ax.axis("off")
    fig2.suptitle("Одинаковый пиксельный размер: сравнимы ли текстуры напрямую")
    fig2.tight_layout()
    fig2.savefig(OUT / "crops_side_by_side.png", dpi=120, bbox_inches="tight")

    fig3, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].hist(own[::8].ravel()[::50], bins=200, color="tab:orange", alpha=0.8)
    axes[0].set_title("свой датасет: гистограмма (uint16)")
    axes[0].set_yscale("log")
    axes[1].hist(sk[::4].ravel()[::50], bins=200, color="tab:blue", alpha=0.8)
    axes[1].set_title("Сколтех Air: гистограмма (uint16)")
    axes[1].set_yscale("log")
    for i in range(0, len(own), 8):
        axes[2].plot(own[i, own.shape[1] // 2], lw=0.6)
    axes[2].set_title("свой: профили яркости по средней строке")
    fig3.tight_layout()
    fig3.savefig(OUT / "histograms.png", dpi=120, bbox_inches="tight")

    with open(OUT / "inspect.md", "w", encoding="utf-8") as f:
        f.write("# Осмотр своего датасета (22322-24)\n\n")
        f.write(f"| | свой | Сколтех |\n|---|---|---|\n")
        f.write(f"| срезов загружено | {len(own)} | 12 |\n")
        f.write(f"| размер среза | {own.shape[1]}x{own.shape[2]} | {sk.shape[1]}x{sk.shape[2]} |\n")
        f.write(f"| min / max | {own.min():.0f} / {own.max():.0f} | {sk.min():.0f} / {sk.max():.0f} |\n")
        f.write(f"| среднее | {own.mean():.0f} | {sk.mean():.0f} |\n")
        f.write(f"| воксель | 1.84 мкм | не указан в датасете |\n")
    print(f"Готово -> {OUT}")


if __name__ == "__main__":
    main()
