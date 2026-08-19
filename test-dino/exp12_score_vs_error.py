"""
Эксперимент 12: связан ли наш скор непохожести с реальной ошибкой реконструкции.

Это та самая проверка, без которой число нельзя называть «доверием к предсказанию».
Команда дала набор ood_testset: 100 срезов, для каждого есть вход модели, её выход
и фактическая ошибка относительно эталона.

Проверяем ТРИ вещи по отдельности, и это важно:

  1. Связь внутри Сколтеха. Один образец, один прибор — «какой это образец»
     не подмешивается. Самая честная из проверок.
  2. Связь внутри университетского керна. То же самое на чужих данных.
  3. Разделение образцов. Здесь как раз возможна ситуация из README набора:
     детектор уверенно кричит «чужое», а предсказание при этом годное.

Целевая метрика — phi_abs_error, промах по интегральной пористости. Именно ради
неё всё делается. mae/rmse/pearson сознательно НЕ берём как основную цель: в этом
наборе они почти целиком определяются шумом эталона, заданным константой на образец
(3.5 против 12.9 п.п.), поэтому «предсказать» их значит просто угадать образец.
Считаем их справочно, чтобы показать этот эффект явно.

Технически: банк построен на срезах 900x900, тестовые — центральные кропы 768x768
тех же данных (проверено побитово). Физический масштаб совпадает, поэтому берём
сетку 2x2 тайла по 224 пикселя из центра кропа.
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

TESTSET = L.PROJECT_ROOT / "ood_testset"
BANK_DIR = L.RUNS_DIR / "bank_full"
OUT = L.RUNS_DIR / "exp12_score_vs_error"
BANK_SLICES, BANK_PER_SLICE, K = 200, 200, 5
TILE, GRID = 224, 2
PROGRESS = OUT / "progress.log"


def log(msg: str):
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build_bank(meta: dict, exclude_z: set[int]) -> torch.Tensor:
    """Банк из срезов Сколтеха. Срезы, попавшие в тест, ИСКЛЮЧАЮТСЯ — иначе
    тестовый срез нашёл бы сам себя и получил бы неправдоподобно хороший скор."""
    air = np.memmap(BANK_DIR / "air_tokens.f16", dtype=np.float16, mode="r",
                    shape=(meta["n_air"], meta["tokens_per_slice"], meta["dim"]))
    allowed = np.array([i for i in range(meta["n_air"]) if i not in exclude_z])
    rng = np.random.default_rng(0)
    ids = rng.permutation(allowed)[:BANK_SLICES]
    vecs = np.concatenate([np.asarray(air[i][:BANK_PER_SLICE], dtype=np.float32) for i in ids], axis=0)
    return F.normalize(torch.from_numpy(vecs), dim=1), ids


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
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")

    meta = json.loads((BANK_DIR / "meta.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(TESTSET / "labels.csv")
    test_z = set(labels[labels["sample"] == "skoltech"]["z"].astype(int))
    log(f"Тестовых срезов: {len(labels)} | из них Сколтеха {len(test_z)} (будут исключены из банка)")

    bank, bank_ids = build_bank(meta, test_z)
    log(f"Банк: {tuple(bank.shape)} из {len(bank_ids)} срезов, пересечений с тестом нет: "
        f"{len(set(bank_ids.tolist()) & test_z) == 0}")

    lo, hi = meta["window"]
    backbone = L.load_dinov2("base", pooling="mean")

    scores = []
    for i, row in labels.iterrows():
        raw = np.array(Image.open(TESTSET / "input_dry" / row["id"])).astype(np.float32)
        # окно яркости Сколтеха применяется к ОБОИМ образцам намеренно: разница
        # шкал и есть сигнал, который детектор должен видеть
        img01 = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
        scores.append(slice_score(backbone.model, img01, bank))
        if (i + 1) % 25 == 0:
            log(f"  {i+1}/{len(labels)} ({time.time()-t0:.0f}s)")
    labels["score"] = scores

    labels.to_csv(OUT / "scored.csv", index=False)
    analyse(labels, time.time() - t0)
    plot(labels)
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    return {"pearson": float(r_p), "p_pearson": float(p_p),
            "spearman": float(r_s), "p_spearman": float(p_s)}


def analyse(d: pd.DataFrame, elapsed: float):
    targets = ["phi_abs_error", "mae", "rmse", "pearson"]
    res = {"within": {}, "between": {}}

    log("\n--- 1-2. Связь ВНУТРИ образца (без подмешивания «какой это образец») ---")
    for s, g in d.groupby("sample"):
        res["within"][s] = {t: corr(g["score"].values, g[t].values) for t in targets}
        log(f"{s} (n={len(g)}):")
        for t in targets:
            c = res["within"][s][t]
            log(f"  скор vs {t:14s} Пирсон {c['pearson']:+.3f} (p={c['p_pearson']:.3f})  "
                f"Спирмен {c['spearman']:+.3f} (p={c['p_spearman']:.3f})")

    log("\n--- 3. Разделение образцов самим скором ---")
    sk = d[d["sample"] == "skoltech"]["score"].values
    un = d[d["sample"] == "university"]["score"].values
    auroc = float(L.auroc(torch.tensor(sk), torch.tensor(un)))
    res["between"] = {"auroc_score": auroc,
                      "score_skoltech": [float(sk.mean()), float(sk.std())],
                      "score_university": [float(un.mean()), float(un.std())]}
    log(f"  скор: Сколтех {sk.mean():.4f}±{sk.std():.4f}, университет {un.mean():.4f}±{un.std():.4f}")
    log(f"  AUROC разделения образцов по скору: {auroc:.3f}")
    for t in targets:
        a = float(L.auroc(torch.tensor(d[d['sample']=='skoltech'][t].values),
                          torch.tensor(d[d['sample']=='university'][t].values)))
        res["between"][f"auroc_{t}"] = a
        log(f"  AUROC разделения по {t:14s}: {a:.3f}")

    log("\n--- Связь на объединённой выборке (осторожно: смешивает образец и ошибку) ---")
    res["pooled"] = {t: corr(d["score"].values, d[t].values) for t in targets}
    for t in targets:
        c = res["pooled"][t]
        log(f"  скор vs {t:14s} Спирмен {c['spearman']:+.3f} (p={c['p_spearman']:.4f})")

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    report(res, d, elapsed)


def report(res, d, elapsed):
    lines = ["# Эксперимент 12: скор непохожести против реальной ошибки модели", "",
             "Проверка, без которой скор нельзя называть «доверием к предсказанию».",
             "Данные: `ood_testset` от команды, модель `2d_unet_l1`.", "",
             "## 1-2. Внутри одного образца", "",
             "Самая честная проверка: один прибор, один керн, «какой это образец» не подмешивается.", "",
             "| Образец | Цель | Пирсон | Спирмен | p (Спирмен) |", "|---|---|---|---|---|"]
    for s in res["within"]:
        for t, c in res["within"][s].items():
            lines.append(f"| {s} | {t} | {c['pearson']:+.3f} | **{c['spearman']:+.3f}** | {c['p_spearman']:.3f} |")

    b = res["between"]
    lines += ["", "## 3. Разделение образцов", "",
              f"- скор Сколтех: {b['score_skoltech'][0]:.4f} ± {b['score_skoltech'][1]:.4f}",
              f"- скор университет: {b['score_university'][0]:.4f} ± {b['score_university'][1]:.4f}",
              f"- **AUROC по нашему скору: {b['auroc_score']:.3f}**", "",
              "| Величина | AUROC разделения образцов |", "|---|---|",
              f"| наш скор | **{b['auroc_score']:.3f}** |"]
    for t in ["phi_abs_error", "mae", "rmse", "pearson"]:
        lines.append(f"| {t} | {b[f'auroc_{t}']:.3f} |")

    lines += ["", "## Как это читать", "",
              "Если наш скор разделяет образцы почти идеально, а `phi_abs_error` — нет,",
              "значит детектор уверенно опознаёт чужие данные, на которых модель тем не",
              "менее считает пористость приемлемо. Тогда трактовать срабатывание как",
              "запрет доверять предсказанию неправильно: полезнее выдавать ожидаемую",
              "величину ошибки, а не бинарный вердикт.", "",
              "`mae`, `rmse` и `pearson` в этом наборе почти целиком определяются шумом",
              "эталона (3.5 п.п. у Сколтеха против 12.9 у университета), заданным",
              "константой на образец. Высокая корреляция с ними означала бы лишь, что",
              "детектор угадал образец, поэтому основная цель — `phi_abs_error`.", "",
              "## Ограничение", "",
              "Точек по оси похожести всего две (свой керн и один чужой), кривую по ним",
              "не построить. Срезы взяты плотно (шаг 4 и 9), поэтому независимых",
              "наблюдений заметно меньше ста.", "",
              f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(d: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"skoltech": "tab:blue", "university": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for s, g in d.groupby("sample"):
        ax.scatter(g["score"], g["phi_abs_error"], s=30, alpha=0.75, c=colors[s], label=s)
    ax.set_xlabel("скор непохожести")
    ax.set_ylabel("промах по пористости, п.п.")
    ax.set_title("Содержательная цель: phi_abs_error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for s, g in d.groupby("sample"):
        ax.scatter(g["score"], g["mae"], s=30, alpha=0.75, c=colors[s], label=s)
    ax.set_xlabel("скор непохожести")
    ax.set_ylabel("MAE φ-карты, п.п.")
    ax.set_title("Лёгкая цель: MAE (определяется шумом эталона)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    data = [d[d["sample"] == s]["score"].values for s in ["skoltech", "university"]]
    ax.boxplot(data, tick_labels=["Сколтех", "университет"])
    ax.set_ylabel("скор непохожести")
    ax.set_title("Разделяет ли скор образцы")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Связан ли наш скор с реальной ошибкой реконструкции")
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_error.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
