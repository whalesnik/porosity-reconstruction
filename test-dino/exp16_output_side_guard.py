"""
Эксперимент 16: сторож на ВЫХОДЕ модели, а не на входе.

До сих пор детектор смотрел на вход: похож ли новый сухой срез на те, что модель
видела при обучении. exp12 и exp15 показали, что этого мало — вход может быть
уверенно опознан как чужой, а предсказание при этом окажется приемлемым.

Здесь проверяется другой подход. Модель предсказывает ксеноновый снимок; значит
можно спросить, похоже ли ЕЁ ПРЕДСКАЗАНИЕ на настоящие ксеноновые снимки. Это
прямая проверка правдоподобности выхода, а не косвенная — по входу. Ровно ради
этого при построении банка (exp09) прогонялись и ксеноновые срезы.

Сравниваются три вещи относительно банка настоящего ксенона Сколтеха:
  1. настоящий ксенон Сколтеха (отложенные срезы) — опорный уровень нормы;
  2. предсказание модели на сколтеховских срезах — модель в своём домене;
  3. предсказание модели на университетском керне — модель за пределами домена.

Если предсказание на чужом керне уходит от многообразия настоящего ксенона
дальше, чем предсказание на своём, у нас появляется сигнал, которого нет на
входной стороне. И главное — проверяется, связан ли этот сигнал с ошибкой.
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

MIXED = L.PROJECT_ROOT / "ood_testset"
UNIV = L.PROJECT_ROOT / "ood_testset_univ"
BANK_DIR = L.RUNS_DIR / "bank_full"
OUT = L.RUNS_DIR / "exp16_output_side_guard"
BANK_SLICES, BANK_PER_SLICE, K = 200, 200, 5
TILE, GRID = 224, 2
PROGRESS = OUT / "progress.log"


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


def own_window(paths) -> tuple[float, float]:
    arr = np.stack([np.array(Image.open(p)).astype(np.float32) for p in paths])
    return L.compute_global_window([arr])


def score_group(model, bank, paths, window) -> list[float]:
    lo, hi = window
    out = []
    for p in paths:
        raw = np.array(Image.open(p)).astype(np.float32)
        img01 = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
        out.append(slice_score(model, img01, bank))
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")

    meta = json.loads((BANK_DIR / "meta.json").read_text(encoding="utf-8"))
    mixed = pd.read_csv(MIXED / "labels.csv")
    univ = pd.read_csv(UNIV / "labels.csv")
    sk_rows = mixed[mixed["sample"] == "skoltech"].reset_index(drop=True)
    sk_test_z = set(sk_rows["z"].astype(int))

    # банк из НАСТОЯЩИХ ксеноновых срезов Сколтеха, тестовые z исключены
    xe = np.memmap(BANK_DIR / "xe_tokens.f16", dtype=np.float16, mode="r",
                   shape=(meta["n_xe"], meta["tokens_per_slice"], meta["dim"]))
    allowed = np.array([i for i in range(meta["n_xe"]) if i not in sk_test_z])
    ids = np.random.default_rng(0).permutation(allowed)[:BANK_SLICES]
    bank = F.normalize(torch.from_numpy(np.concatenate(
        [np.asarray(xe[i][:BANK_PER_SLICE], dtype=np.float32) for i in ids], axis=0)), dim=1)
    log(f"Банк настоящего ксенона: {tuple(bank.shape)} | пересечений с тестом: "
        f"{len(set(ids.tolist()) & sk_test_z)}")

    backbone = L.load_dinov2("base", pooling="mean")
    sk_win = tuple(meta["window"])

    # 1. опорный уровень: настоящий ксенон Сколтеха на отложенных z
    xe_files = L.list_slice_files(L.XE_DIR, "Xe")
    real_paths = [xe_files[z] for z in sorted(sk_test_z)]
    log("\n1) настоящий ксенон Сколтеха (опорный уровень)...")
    real = score_group(backbone.model, bank, real_paths, sk_win)
    log(f"   {np.mean(real):.4f} +- {np.std(real):.4f}  [{min(real):.4f}; {max(real):.4f}]")

    # 2. предсказание модели на сколтеховских срезах — свой домен
    log("2) предсказание модели, сколтеховские срезы...")
    sk_pred_paths = [MIXED / "pred_xenon" / i for i in sk_rows["id"]]
    sk_pred = score_group(backbone.model, bank, sk_pred_paths, sk_win)
    log(f"   {np.mean(sk_pred):.4f} +- {np.std(sk_pred):.4f}  [{min(sk_pred):.4f}; {max(sk_pred):.4f}]")

    # 3. предсказание модели на университетском керне — чужой домен.
    # Две нормировки: окном Сколтеха (видна разница шкал) и собственным окном
    # предсказаний (разница шкал убрана, остаётся структура).
    log("3) предсказание модели, университетский керн...")
    un_pred_paths = [UNIV / "pred_xenon" / i for i in univ["id"]]
    un_own = own_window(un_pred_paths)
    log(f"   собственное окно предсказаний: {un_own[0]:.0f}..{un_own[1]:.0f}")
    un_pred_sk = score_group(backbone.model, bank, un_pred_paths, sk_win)
    un_pred_own = score_group(backbone.model, bank, un_pred_paths, un_own)
    log(f"   окно Сколтеха:      {np.mean(un_pred_sk):.4f} +- {np.std(un_pred_sk):.4f}")
    log(f"   собственное окно:   {np.mean(un_pred_own):.4f} +- {np.std(un_pred_own):.4f}")

    sk_rows = sk_rows.assign(out_score=sk_pred)
    univ = univ.assign(out_score_skwin=un_pred_sk, out_score_ownwin=un_pred_own)
    sk_rows.to_csv(OUT / "scored_skoltech.csv", index=False)
    univ.to_csv(OUT / "scored_univ.csv", index=False)

    res = analyse(real, sk_rows, univ, time.time() - t0)
    plot(real, sk_rows, univ)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def analyse(real, sk, un, elapsed):
    from scipy.stats import spearmanr
    res = {"levels": {"real_xe": [float(np.mean(real)), float(np.std(real))],
                      "pred_skoltech": [float(sk.out_score.mean()), float(sk.out_score.std())],
                      "pred_univ_skwin": [float(un.out_score_skwin.mean()), float(un.out_score_skwin.std())],
                      "pred_univ_ownwin": [float(un.out_score_ownwin.mean()), float(un.out_score_ownwin.std())]}}

    log("\n--- Насколько предсказание похоже на настоящий ксенон ---")
    for k, v in res["levels"].items():
        log(f"  {k:20s} {v[0]:.4f} +- {v[1]:.4f}")

    auroc_pred = float(L.auroc(torch.tensor(sk.out_score.values),
                               torch.tensor(un.out_score_ownwin.values)))
    auroc_real = float(L.auroc(torch.tensor(np.array(real)),
                               torch.tensor(sk.out_score.values)))
    res["auroc_pred_sk_vs_univ_ownwin"] = auroc_pred
    res["auroc_realxe_vs_pred_sk"] = auroc_real
    log(f"\n  AUROC: предсказание своё против чужого (собств. окно) = {auroc_pred:.3f}")
    log(f"  AUROC: настоящий ксенон против предсказания на своём   = {auroc_real:.3f}")

    log("\n--- Связь выходного скора с ошибкой ---")
    res["corr"] = {}
    for name, df, col in [("skoltech", sk, "out_score"),
                          ("univ (окно Сколтеха)", un, "out_score_skwin"),
                          ("univ (собств. окно)", un, "out_score_ownwin")]:
        res["corr"][name] = {}
        targets = [c for c in ["phi_abs_error", "phi_abs_error_debiased", "mae"] if c in df.columns]
        for t in targets:
            r = spearmanr(df[col].values, df[t].values)
            res["corr"][name][t] = {"spearman": float(r.statistic), "p": float(r.pvalue)}
            log(f"  {name:22s} ~ {t:26s} rho={r.statistic:+.3f} (p={r.pvalue:.3f})")

    report(res, elapsed)
    return res


def report(res, elapsed):
    lv = res["levels"]
    lines = ["# Эксперимент 16: сторож на выходе модели", "",
             "Проверяется не вход, а выход: похоже ли предсказание модели на настоящие",
             "ксеноновые снимки. Банк — 200 срезов настоящего ксенона Сколтеха,",
             "тестовые срезы из него исключены.", "",
             "## Расстояние до многообразия настоящего ксенона", "",
             "| Что подано | скор | std |", "|---|---|---|",
             f"| настоящий ксенон Сколтеха (опора) | {lv['real_xe'][0]:.4f} | {lv['real_xe'][1]:.4f} |",
             f"| предсказание, сколтеховские срезы | {lv['pred_skoltech'][0]:.4f} | {lv['pred_skoltech'][1]:.4f} |",
             f"| предсказание, университет (окно Сколтеха) | {lv['pred_univ_skwin'][0]:.4f} | {lv['pred_univ_skwin'][1]:.4f} |",
             f"| предсказание, университет (собств. окно) | {lv['pred_univ_ownwin'][0]:.4f} | {lv['pred_univ_ownwin'][1]:.4f} |",
             "",
             f"- AUROC, предсказание своё против чужого (собств. окно): **{res['auroc_pred_sk_vs_univ_ownwin']:.3f}**",
             f"- AUROC, настоящий ксенон против предсказания на своём: {res['auroc_realxe_vs_pred_sk']:.3f}",
             "", "## Связь выходного скора с ошибкой", "",
             "| Группа | Цель | rho | p |", "|---|---|---|---|"]
    for grp, d in res["corr"].items():
        for t, c in d.items():
            lines.append(f"| {grp} | `{t}` | {c['spearman']:+.3f} | {c['p']:.3f} |")
    lines += ["", f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(real, sk, un):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.boxplot([real, sk.out_score.values, un.out_score_ownwin.values, un.out_score_skwin.values],
               tick_labels=["настоящий\nксенон", "предсказание\nСколтех",
                            "предсказание\nуниверситет\n(своё окно)",
                            "предсказание\nуниверситет\n(окно Ск.)"])
    ax.set_ylabel("расстояние до банка настоящего ксенона")
    ax.set_title("Правдоподобность выхода модели")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.scatter(sk.out_score, sk.phi_abs_error, s=28, c="tab:blue", label="Сколтех")
    ax.scatter(un.out_score_ownwin, un.phi_abs_error, s=28, c="tab:red", label="университет")
    ax.set_xlabel("выходной скор")
    ax.set_ylabel("phi_abs_error")
    ax.set_title("Выходной скор против ошибки по пористости")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(un.out_score_ownwin, un.phi_abs_error_debiased, s=28, c="tab:red")
    ax.set_xlabel("выходной скор (университет, своё окно)")
    ax.set_ylabel("phi_abs_error_debiased")
    ax.set_title("После вычитания смещения")
    ax.grid(alpha=0.3)

    fig.suptitle("Сторож на выходе: похоже ли предсказание на настоящий ксенон")
    fig.tight_layout()
    fig.savefig(OUT / "output_side_guard.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
