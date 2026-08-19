"""
Эксперимент 15: детектор против ошибки модели на университетском керне.

Набор ood_testset_univ: 100 срезов образца 22322-24, равномерно по всему объёму
(z 5..469), модель 2d_unet_l1 не видела ни одного вокселя этого керна.

Чем этот набор ценнее прежнего ood_testset. Там университетских срезов было 50 и
ошибка считалась «как есть», то есть вместе с постоянным смещением модели, которое
доминировало и делало связь непроверяемой. Здесь приложена ошибка после вычитания
смещения — переменная часть, которую детектор в принципе может предсказывать.

Что проверяется:
  1. Связь скора с ошибкой внутри одного образца, по трём версиям ошибки
     (как есть / поправка со Сколтеха / поправка-оракул).
  2. Поправка на z: в этом объёме есть известный дрейф калибровки вдоль высоты
     (оговорён в README набора), поэтому связь может оказаться наведённой.
  3. Насыщен ли скор. В exp12 на университетских срезах он стоял в узкой полосе
     0.323 +- 0.015 — если так же, ранжировать ошибку он не сможет по построению.

Скор считается в двух нормировках: окном Сколтеха (как в exp12) и собственным
окном образца (убирает разницу шкал, оставляет структуру).
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

SET = L.PROJECT_ROOT / "ood_testset_univ"
BANK_DIR = L.RUNS_DIR / "bank_full"
OUT = L.RUNS_DIR / "exp15_univ_score_vs_error"
BANK_SLICES, BANK_PER_SLICE, K = 200, 200, 5
TILE, GRID = 224, 2
PROGRESS = OUT / "progress.log"

TARGETS = ["phi_abs_error", "phi_abs_error_debiased",
           "phi_abs_error_debiased_oracle", "mae", "rmse", "pearson"]


def log(msg: str):
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


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
    dist = 1 - q @ bank.T
    return float(dist.topk(K, largest=False).values.mean(dim=1).mean())


def partial_spearman(x, y, ctrl):
    """Ранговая частная корреляция: убираем линейный тренд по ctrl из обоих рядов."""
    from scipy.stats import spearmanr, rankdata
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(ctrl)

    def res(a):
        return a - np.polyval(np.polyfit(rc, a, 1), rc)

    return spearmanr(res(rx), res(ry))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")

    meta = json.loads((BANK_DIR / "meta.json").read_text(encoding="utf-8"))
    d = pd.read_csv(SET / "labels.csv")
    log(f"Срезов: {len(d)} | z {d.z.min():.0f}..{d.z.max():.0f} | воксель {d.voxel_um.iloc[0]:.3f} мкм")

    air = np.memmap(BANK_DIR / "air_tokens.f16", dtype=np.float16, mode="r",
                    shape=(meta["n_air"], meta["tokens_per_slice"], meta["dim"]))
    ids = np.random.default_rng(0).permutation(meta["n_air"])[:BANK_SLICES]
    bank = F.normalize(torch.from_numpy(np.concatenate(
        [np.asarray(air[i][:BANK_PER_SLICE], dtype=np.float32) for i in ids], axis=0)), dim=1)
    log(f"Банк Сколтеха: {tuple(bank.shape)} из {BANK_SLICES} срезов")

    backbone = L.load_dinov2("base", pooling="mean")
    sk_lo, sk_hi = meta["window"]

    raws = []
    for i, row in d.iterrows():
        raws.append(np.array(Image.open(SET / "input_dry" / row["id"])).astype(np.float32))
    own_lo, own_hi = L.compute_global_window([np.stack(raws)])
    log(f"Окно Сколтеха: {sk_lo:.0f}..{sk_hi:.0f} | собственное окно образца: {own_lo:.0f}..{own_hi:.0f}")

    for tag, (lo, hi) in {"sk_window": (sk_lo, sk_hi), "own_window": (own_lo, own_hi)}.items():
        s = []
        for i, raw in enumerate(raws):
            img01 = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
            s.append(slice_score(backbone.model, img01, bank))
            if (i + 1) % 50 == 0:
                log(f"  {tag}: {i+1}/{len(raws)} ({time.time()-t0:.0f}s)")
        d[f"score_{tag}"] = s
        log(f"  {tag}: {np.mean(s):.4f} +- {np.std(s):.4f}  [{min(s):.4f}; {max(s):.4f}]")

    d.to_csv(OUT / "scored.csv", index=False)
    analyse(d, time.time() - t0)
    plot(d)
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def analyse(d: pd.DataFrame, elapsed: float):
    from scipy.stats import spearmanr
    res = {}
    for tag in ["sk_window", "own_window"]:
        sc = d[f"score_{tag}"].values
        log(f"\n--- Скор ({tag}) против ошибки, внутри образца ---")
        res[tag] = {}
        for t in TARGETS:
            raw = spearmanr(sc, d[t].values)
            par = partial_spearman(sc, d[t].values, d["z"].values)
            res[tag][t] = {"spearman": float(raw.statistic), "p": float(raw.pvalue),
                           "partial_z": float(par.statistic), "p_partial": float(par.pvalue)}
            log(f"  {t:32s} rho={raw.statistic:+.3f} (p={raw.pvalue:.3f})   "
                f"с поправкой на z: {par.statistic:+.3f} (p={par.pvalue:.3f})")
        rz = spearmanr(sc, d["z"].values)
        res[tag]["_score_vs_z"] = {"spearman": float(rz.statistic), "p": float(rz.pvalue)}
        res[tag]["_stats"] = {"mean": float(np.mean(sc)), "std": float(np.std(sc)),
                              "min": float(np.min(sc)), "max": float(np.max(sc))}
        log(f"  {'скор ~ z':32s} rho={rz.statistic:+.3f} (p={rz.pvalue:.3f})")

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    report(res, d, elapsed)


def report(res, d, elapsed):
    lines = ["# Эксперимент 15: детектор против ошибки модели на университетском керне", "",
             "Набор `ood_testset_univ`: 100 срезов образца 22322-24 (z 5…469), модель",
             "`2d_unet_l1` этого керна не видела. Отличие от прежнего набора — приложена",
             "ошибка после вычитания систематического смещения, то есть переменная часть,",
             "которую детектор в принципе может предсказывать.", "",
             "## Разброс скора", "",
             "| Нормировка | среднее | std | размах |", "|---|---|---|---|"]
    for tag in ["sk_window", "own_window"]:
        s = res[tag]["_stats"]
        lines.append(f"| {tag} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} … {s['max']:.4f} |")

    lines += ["", "## Связь скора с ошибкой (ранговая корреляция, n = 100)", "",
              "Поправка на `z` нужна потому, что в этом объёме есть известный дрейф",
              "калибровки вдоль высоты (оговорён в README набора): без неё связь может",
              "оказаться наведённой общим трендом по глубине.", "",
              "| Нормировка | Цель | rho | p | rho с поправкой на z | p |",
              "|---|---|---|---|---|---|"]
    for tag in ["sk_window", "own_window"]:
        for t in TARGETS:
            c = res[tag][t]
            lines.append(f"| {tag} | `{t}` | {c['spearman']:+.3f} | {c['p']:.3f} | "
                         f"**{c['partial_z']:+.3f}** | {c['p_partial']:.3f} |")
    lines += ["", f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    pairs = [("phi_abs_error", "ошибка как есть"),
             ("phi_abs_error_debiased", "ошибка после поправки"),
             ("mae", "локальная MAE")]
    for j, (col, title) in enumerate(pairs):
        for i, tag in enumerate(["sk_window", "own_window"]):
            ax = axes[i, j]
            sc = ax.scatter(d[f"score_{tag}"], d[col], c=d["z"], cmap="viridis", s=28)
            ax.set_xlabel(f"скор ({tag})")
            ax.set_ylabel(col)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            if j == 2:
                plt.colorbar(sc, ax=ax, label="z")
    fig.suptitle("Университетский керн: связан ли скор с ошибкой модели (цвет — глубина)")
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_error_univ.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
