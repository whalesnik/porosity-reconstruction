"""
Эксперимент 13: генерация набора для построения калибровочной кривой.

Зачем. exp12 показал, что связь «непохожесть -> ошибка» на наборе ood_testset
проверить как следует нельзя: там всего две точки по оси похожести (свой керн и
один чужой), а кривую по двум точкам не построить.

Решение, не требующее новых сканов: берём сколтеховские срезы, портим ВХОД модели
искажениями заданной силы и прогоняем модель заново. Сила искажения известна,
эталон не меняется, точек получается столько, сколько нужно.

Ключевая тонкость, от которой зависит осмысленность всего набора.
Искажается ТОЛЬКО сухой вход. Порода при этом не меняется — меняется качество её
измерения. Значит истинная пористость остаётся той же, что у чистого среза, и
ошибка считается относительно неё:

    ошибка = | phi_pred(искажённый вход) - phi_real(чистый срез) |

Если пересчитывать «истину» из искажённых данных, эксперимент теряет смысл.

Базовые срезы берутся из ood_testset, поэтому phi_real, beta, d100 и константы
нормировки уже известны и переносятся в манифест. Коллегам остаётся только
прогнать модель и посчитать phi_pred — пересчитывать эталон не нужно.

Искажения применяются в тех единицах, в которых модель видит вход:
y = (raw - offset) / scale. Так сила искажения напрямую сопоставима с масштабом,
на который настроена сеть.
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

import lib_dino as L

TESTSET = L.PROJECT_ROOT / "ood_testset"
OUT = L.PROJECT_ROOT / "calibration_set"
N_BASE = 15
SEVERITIES = [1, 2, 3, 4, 5]

# Уровни силы — те же, что в наших экспериментах (lib_dino.CORRUPTIONS), но
# применяются без обрезки в [0,1]: обрезка съела бы яркие минеральные включения.
BRIGHTNESS = [0.02, 0.05, 0.10, 0.20, 0.35]
CONTRAST = [0.9, 0.75, 0.6, 0.45, 0.3]
NOISE = [0.01, 0.02, 0.04, 0.08, 0.15]
BLUR = [0.5, 1.0, 2.0, 3.5, 6.0]
RESOLUTION = [1.5, 2.0, 3.0, 4.0, 6.0]


def log(msg: str):
    print(msg)
    sys.stdout.flush()


def c_brightness(y, s, rng=None):
    return y + BRIGHTNESS[s - 1]


def c_contrast(y, s, rng=None):
    return (y - y.mean()) * CONTRAST[s - 1] + y.mean()


def c_noise(y, s, rng=None):
    return y + rng.normal(0, NOISE[s - 1], y.shape).astype(np.float32)


def c_blur(y, s, rng=None):
    return gaussian_filter(y, BLUR[s - 1])


def c_resolution(y, s, rng=None):
    """Имитация худшего разрешения прибора — физически самое осмысленное
    искажение для задачи про субразрешённые поры."""
    f = RESOLUTION[s - 1]
    x = torch.from_numpy(y).float()[None, None]
    small = F.interpolate(x, scale_factor=1 / f, mode="area")
    return F.interpolate(small, size=y.shape, mode="bilinear", align_corners=False)[0, 0].numpy()


CORRUPTIONS = {"brightness": (c_brightness, BRIGHTNESS),
               "contrast": (c_contrast, CONTRAST),
               "noise": (c_noise, NOISE),
               "blur": (c_blur, BLUR),
               "resolution": (c_resolution, RESOLUTION)}


def main():
    t0 = time.time()
    (OUT / "input_dry").mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(TESTSET / "labels.csv")
    sk = labels[labels["sample"] == "skoltech"].sort_values("z").reset_index(drop=True)
    # базовые срезы равномерно по отложенному диапазону z 720..899
    base_idx = np.linspace(0, len(sk) - 1, N_BASE, dtype=int)
    bases = sk.iloc[base_idx].reset_index(drop=True)
    log(f"Базовых срезов: {len(bases)}, z от {bases.z.min()} до {bases.z.max()}")

    rows, n = [], 0
    for _, b in bases.iterrows():
        raw = np.array(Image.open(TESTSET / "input_dry" / b["id"])).astype(np.float32)
        y0 = (raw - b["offset"]) / b["scale"]

        variants = [("clean", 0, y0)]
        for cname, (fn, levels) in CORRUPTIONS.items():
            for s in SEVERITIES:
                rng = np.random.default_rng(1000 * int(b["z"]) + 10 * s + len(cname))
                variants.append((cname, s, fn(y0, s, rng)))

        for cname, s, y in variants:
            out = np.clip(y * b["scale"] + b["offset"], 0, 65535).round().astype(np.uint16)
            fid = f"img_{n:04d}.png"
            Image.fromarray(out).save(OUT / "input_dry" / fid)
            level = 0.0 if cname == "clean" else float(CORRUPTIONS[cname][1][s - 1])
            rows.append({"id": fid, "base_id": b["id"], "z": int(b["z"]),
                         "corruption": cname, "severity": s, "level": level,
                         "phi_real": b["phi_real"], "voxel_um": b["voxel_um"],
                         "offset": b["offset"], "scale": b["scale"],
                         "beta": b["beta"], "d100": b["d100"],
                         "sigma_phi_noise": b["sigma_phi_noise"]})
            n += 1
        log(f"  z={int(b['z'])}: {len(variants)} вариантов, всего {n}")

    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    log(f"\nСохранено {n} изображений, манифест на {len(man)} строк")

    template = man[["id", "base_id", "corruption", "severity"]].copy()
    for c in ["phi_pred", "mae", "rmse", "pearson"]:
        template[c] = ""
    template.to_csv(OUT / "results_template.csv", index=False)

    preview(bases.iloc[len(bases) // 2], man)
    log(f"Готово за {time.time()-t0:.0f}s -> {OUT}")


def preview(base, man):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = man[man["base_id"] == base["id"]]
    lo, hi = base["offset"], base["offset"] + base["scale"]
    fig, axes = plt.subplots(len(CORRUPTIONS), 6, figsize=(17, 14))
    for i, cname in enumerate(CORRUPTIONS):
        clean = sub[sub["corruption"] == "clean"].iloc[0]
        img = np.array(Image.open(OUT / "input_dry" / clean["id"]))
        axes[i, 0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        axes[i, 0].set_title("severity 0 (чистый)" if i == 0 else "", fontsize=9)
        axes[i, 0].set_ylabel(cname, fontsize=11)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        for s in SEVERITIES:
            r = sub[(sub["corruption"] == cname) & (sub["severity"] == s)].iloc[0]
            img = np.array(Image.open(OUT / "input_dry" / r["id"]))
            ax = axes[i, s]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"severity {s} ({r['level']:g})" if i == 0 else f"{r['level']:g}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Искажения входа, единое окно яркости. Базовый срез z={int(base['z'])}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "preview.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
