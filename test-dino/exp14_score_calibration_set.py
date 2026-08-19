"""
Эксперимент 14: посчитать скор непохожести для всех срезов calibration_set.

Скор считается заранее и кладётся прямо в манифест, чтобы коллегам с обученной
моделью осталось вернуть только ошибки — сопоставление они смогут сделать сами,
не запуская наш код.

Банк тот же, что в exp12: 200 срезов Сколтеха, из которых исключены все z,
использованные как базовые в calibration_set. Без этого исключения базовый срез
находил бы сам себя и получал неправдоподобно низкий скор, а вся калибровочная
кривая оказалась бы смещена в своей левой, самой важной точке.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import lib_dino as L

CALSET = L.PROJECT_ROOT / "calibration_set"
BANK_DIR = L.RUNS_DIR / "bank_full"
BANK_SLICES, BANK_PER_SLICE, K = 200, 200, 5
TILE, GRID = 224, 2


def log(msg: str):
    print(msg)
    sys.stdout.flush()


@torch.no_grad()
def slice_score(model, img01: np.ndarray, bank: torch.Tensor) -> float:
    h, w = img01.shape
    span = TILE * GRID
    r0, c0 = (h - span) // 2, (w - span) // 2
    tiles = [img01[r0 + i * TILE:r0 + (i + 1) * TILE, c0 + j * TILE:c0 + (j + 1) * TILE]
             for i in range(GRID) for j in range(GRID)]
    x = torch.cat([L.to_tensor(t) for t in tiles], dim=0)
    tok = model(pixel_values=x).last_hidden_state[:, 1:, :]
    q = F.normalize(tok.reshape(-1, tok.shape[-1]), dim=1)
    d = 1 - q @ bank.T
    return float(d.topk(K, largest=False).values.mean(dim=1).mean())


def main():
    t0 = time.time()
    meta = json.loads((BANK_DIR / "meta.json").read_text(encoding="utf-8"))
    man = pd.read_csv(CALSET / "manifest.csv")
    exclude = set(man["z"].astype(int))
    log(f"Срезов в наборе: {len(man)} | базовых z исключено из банка: {len(exclude)}")

    air = np.memmap(BANK_DIR / "air_tokens.f16", dtype=np.float16, mode="r",
                    shape=(meta["n_air"], meta["tokens_per_slice"], meta["dim"]))
    allowed = np.array([i for i in range(meta["n_air"]) if i not in exclude])
    ids = np.random.default_rng(0).permutation(allowed)[:BANK_SLICES]
    bank = F.normalize(torch.from_numpy(np.concatenate(
        [np.asarray(air[i][:BANK_PER_SLICE], dtype=np.float32) for i in ids], axis=0)), dim=1)
    log(f"Банк: {tuple(bank.shape)} | пересечений с базовыми: {len(set(ids.tolist()) & exclude)}")

    lo, hi = meta["window"]
    backbone = L.load_dinov2("base", pooling="mean")
    scores = []
    for i, row in man.iterrows():
        raw = np.array(Image.open(CALSET / "input_dry" / row["id"])).astype(np.float32)
        img01 = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
        scores.append(slice_score(backbone.model, img01, bank))
        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(man)} ({time.time()-t0:.0f}s)")
    man["ood_score"] = scores
    man.to_csv(CALSET / "manifest.csv", index=False)

    log("\nСкор по уровням искажения (медиана):")
    piv = man.pivot_table(index="corruption", columns="severity", values="ood_score", aggfunc="median")
    log(piv.round(4).to_string())
    base = man[man.corruption == "clean"]["ood_score"].median()
    log(f"\nчистые срезы: {base:.4f}")
    log(f"диапазон по всему набору: {man.ood_score.min():.4f} .. {man.ood_score.max():.4f}")
    plot(man, base)
    log(f"\nГотово за {time.time()-t0:.0f}s")


def plot(man, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, g in man[man.corruption != "clean"].groupby("corruption"):
        m = g.groupby("severity")["ood_score"].median()
        ax.plot([0] + list(m.index), [base] + list(m.values), "o-", label=c)
    ax.axhline(base, color="gray", ls=":", lw=1)
    ax.set_xlabel("сила искажения (severity)")
    ax.set_ylabel("скор непохожести")
    ax.set_title("Скор растёт вместе с силой искажения — ось для калибровочной кривой")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CALSET / "score_vs_severity.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
