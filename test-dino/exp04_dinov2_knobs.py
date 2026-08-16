"""
Эксперимент 4: перебор внутренних ручек DINOv2 на градуированном бенчмарке.

Мотивация. В exp02 DINOv2 проиграла 11-мерной классической статистике
(0.66 против 0.93 среднего AUROC). Две конкурирующие гипотезы:

  H1 "виновато разрешение": срез 900x900 сжимается до 224x224, мелкая текстура
     пор физически уничтожается до того, как модель что-то увидит.
  H2 "виновата инвариантность": DINOv2 обучена быть устойчивой к яркости,
     контрасту и шуму — ровно к тем артефактам съёмки, которые мы детектируем.
     Тогда никакая настройка не поможет, дело в самой предобученной модели.

Эти гипотезы различимы: если H1 — переход к нативному разрешению (crop/tiles)
и к 448 должен резко поднять AUROC. Если H2 — не поднимет ничего.

Перебираем: вход (resize/crop/tiles x 224/448) x пулинг (cls/mean/cls+mean/mean+std)
x слой (последний/средний) x скор (kNN-cos/Махаланобис). Бэкбон — dinov2-small
ради скорости, лучшие конфигурации перепроверяем на dinov2-base.

Опорная линия сравнения — classic-stats из exp02 (0.927).
"""

from __future__ import annotations

import json
import time
import numpy as np
import torch
from itertools import product

import lib_dino as L

N_REF, N_TEST, N_XE = 40, 20, 20
SEVERITIES = [1, 2, 3, 4, 5]
CLASSIC_BASELINE = 0.927  # classic-stats/identity из exp02 — то, что нужно побить
OUT = L.RUNS_DIR / "exp04_dinov2_knobs"


def build_input_variants(vol01: np.ndarray, crop_box, rng) -> dict[str, list[np.ndarray]]:
    """Один срез -> несколько представлений входа. Список, т.к. tiles даёт несколько
    картинок на срез (эмбеддинги потом усредняются)."""
    return {
        "resize224": [L.prepare_2d(vol01, 224, "resize")],
        "resize448": [L.prepare_2d(vol01, 448, "resize")],
        "crop224": [L.prepare_2d(vol01, 224, "crop", crop_box)],
        "crop448": [L.prepare_2d(vol01, 448, "crop", crop_box)],
        "tiles4x224": L.tile_crops(vol01, tile=224, n_tiles=4, rng=rng),
    }


@torch.no_grad()
def embed_sets(backbone, prepared: dict[str, list[list[np.ndarray]]]) -> dict[str, torch.Tensor]:
    """prepared[set_name] = список срезов, каждый — список тайлов.
    Эмбеддинг среза = среднее по его тайлам (для не-tiles режимов тайл один)."""
    out = {}
    for set_name, slices in prepared.items():
        vecs = []
        for tiles in slices:
            t = L.embed_batch(backbone, [L.to_tensor(x) for x in tiles])
            vecs.append(t.mean(dim=0, keepdim=True))
        out[set_name] = torch.cat(vecs, dim=0).float()
    return out


def evaluate(feats: dict[str, torch.Tensor], score_name: str) -> dict:
    ref = feats["__ref__"]
    score_fn = ((lambda q: L.knn_score(q, ref, k=5, metric="cosine"))
                if score_name == "knn5_cos" else
                (lambda q: L.mahalanobis_score(q, ref)))
    s_clean = score_fn(feats["clean"])
    per_corruption, xe_auroc = {}, None
    for set_name, f in feats.items():
        if set_name in ("__ref__", "clean"):
            continue
        a = L.auroc(s_clean, score_fn(f))
        if set_name == "xe_natural":
            xe_auroc = a
        else:
            per_corruption.setdefault(set_name.rsplit("_s", 1)[0], []).append(a)
    means = {k: float(np.mean(v)) for k, v in per_corruption.items()}
    return {"per_corruption": means,
            "mean_synthetic": float(np.mean(list(means.values()))),
            "xe_natural": xe_auroc}


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Загружаю срезы...")
    air_all, _ = L.load_air(N_REF + N_TEST)
    xe_all, _ = L.load_xe(N_XE)
    g_lo, g_hi = L.compute_global_window([air_all])
    phys = np.stack([np.clip((v - g_lo) / (g_hi - g_lo), 0, 1).astype(np.float32) for v in air_all])
    perm = rng.permutation(len(phys))
    ref_raw, test_raw = phys[perm[:N_REF]], phys[perm[N_REF:]]
    xe_phys = np.stack([np.clip((v - g_lo) / (g_hi - g_lo), 0, 1).astype(np.float32) for v in xe_all])
    crop_box = L.center_box(air_all[0].shape)

    print("Готовлю искажённые наборы...")
    sets = {"__ref__": ref_raw, "clean": test_raw, "xe_natural": xe_phys}
    for cname, cfn in L.CORRUPTIONS.items():
        for sev in SEVERITIES:
            kw = {"rng": np.random.default_rng(sev)} if cname == "noise" else {}
            sets[f"{cname}_s{sev}"] = np.stack([cfn(v, sev, **kw) for v in test_raw])

    # предподготовка всех вариантов входа один раз — самая дорогая часть по CPU
    print("Готовлю варианты входа (5 режимов)...")
    input_names = list(build_input_variants(ref_raw[0], crop_box, np.random.default_rng(0)).keys())
    prepared_by_input = {name: {} for name in input_names}
    for set_name, vols in sets.items():
        for i, v in enumerate(vols):
            variants = build_input_variants(v, crop_box, np.random.default_rng(1000 + i))
            for name in input_names:
                prepared_by_input[name].setdefault(set_name, []).append(variants[name])

    rows = []
    # --- фаза 1: dinov2-small, полный перебор входа x пулинга x скора (слой последний) ---
    poolings = ["cls", "mean", "cls+mean", "mean+std"]
    print(f"\nФаза 1: dinov2-small, {len(input_names)}x{len(poolings)} конфигураций")
    for pooling in poolings:
        bb = L.load_dinov2("small", pooling=pooling)
        for inp in input_names:
            t = time.time()
            feats = embed_sets(bb, prepared_by_input[inp])
            for score in ("knn5_cos", "mahalanobis"):
                r = evaluate(feats, score)
                rows.append({"model": "small", "layer": "last", "pooling": pooling,
                             "input": inp, "score": score, **r})
            best = max(r["mean_synthetic"] for r in rows[-2:])
            print(f"  small {pooling:9s} {inp:11s} лучший mean_synthetic={best:.3f}  ({time.time()-t:.0f}s)")

    # --- фаза 2: лучший (вход, пулинг) — проверяем средние слои ---
    best_row = max(rows, key=lambda r: r["mean_synthetic"])
    print(f"\nФаза 2: слои. Лучшее из фазы 1: {best_row['pooling']}/{best_row['input']} "
          f"= {best_row['mean_synthetic']:.3f}")
    for layer in (3, 6, 9):
        bb = L.load_dinov2("small", pooling=best_row["pooling"], layer=layer)
        feats = embed_sets(bb, prepared_by_input[best_row["input"]])
        for score in ("knn5_cos", "mahalanobis"):
            r = evaluate(feats, score)
            rows.append({"model": "small", "layer": f"L{layer}", "pooling": best_row["pooling"],
                         "input": best_row["input"], "score": score, **r})
        print(f"  small L{layer}: {max(r['mean_synthetic'] for r in rows[-2:]):.3f}")

    # --- фаза 3: лучшая конфигурация целиком — перепроверка на dinov2-base ---
    best_row = max(rows, key=lambda r: r["mean_synthetic"])
    layer = None if best_row["layer"] == "last" else int(best_row["layer"][1:])
    print(f"\nФаза 3: base с лучшей конфигурацией {best_row['pooling']}/"
          f"{best_row['input']}/{best_row['layer']}")
    bb = L.load_dinov2("base", pooling=best_row["pooling"], layer=layer)
    feats = embed_sets(bb, prepared_by_input[best_row["input"]])
    for score in ("knn5_cos", "mahalanobis"):
        r = evaluate(feats, score)
        rows.append({"model": "base", "layer": best_row["layer"], "pooling": best_row["pooling"],
                     "input": best_row["input"], "score": score, **r})
        print(f"  base {score}: {r['mean_synthetic']:.3f}")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(rows, time.time() - t0)
    plot(rows)
    print(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def write_report(rows: list[dict], elapsed: float):
    rows_sorted = sorted(rows, key=lambda r: -r["mean_synthetic"])
    lines = ["# Эксперимент 4: перебор ручек DINOv2", "",
             f"Опорная точка — classic-stats из exp02: **{CLASSIC_BASELINE:.3f}**.",
             "Вопрос эксперимента: можно ли настройкой DINOv2 побить простую статистику.", "",
             "| Модель | Слой | Пулинг | Вход | Скор | Синтетика | Xe |",
             "|---|---|---|---|---|---|---|"]
    for r in rows_sorted[:20]:
        lines.append(f"| {r['model']} | {r['layer']} | {r['pooling']} | {r['input']} | "
                     f"{r['score']} | **{r['mean_synthetic']:.3f}** | {r['xe_natural']:.3f} |")
    best = rows_sorted[0]
    lines += ["", "## Итог", "",
              f"- Лучшая конфигурация DINOv2: **{best['mean_synthetic']:.3f}** "
              f"({best['model']}/{best['layer']}/{best['pooling']}/{best['input']}/{best['score']})",
              f"- Классическая статистика: **{CLASSIC_BASELINE:.3f}**",
              f"- Разрыв: **{best['mean_synthetic'] - CLASSIC_BASELINE:+.3f}**", ""]
    by_input = {}
    for r in rows:
        by_input.setdefault(r["input"], []).append(r["mean_synthetic"])
    lines += ["## Проверка гипотезы H1 (виновато разрешение)", "",
              "| Вход | Лучший AUROC |", "|---|---|"]
    for k, v in sorted(by_input.items(), key=lambda kv: -max(kv[1])):
        lines.append(f"| {k} | {max(v):.3f} |")
    lines.append("")
    lines.append(f"_Время: {elapsed:.0f}s_")
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    inputs = sorted({r["input"] for r in rows})
    poolings = sorted({r["pooling"] for r in rows})

    ax = axes[0]
    width = 0.8 / len(poolings)
    for j, p in enumerate(poolings):
        ys = [max([r["mean_synthetic"] for r in rows
                   if r["input"] == i and r["pooling"] == p] or [0]) for i in inputs]
        ax.bar(np.arange(len(inputs)) + j * width, ys, width, label=p)
    ax.axhline(CLASSIC_BASELINE, color="crimson", ls="--", lw=2, label="classic-stats (exp02)")
    ax.set_xticks(np.arange(len(inputs)) + 0.4 - width / 2)
    ax.set_xticklabels(inputs, rotation=20)
    ax.set_ylabel("средний AUROC по искажениям")
    ax.set_title("Разрешение и пулинг против простой статистики")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    layers = ["last", "L3", "L6", "L9"]
    ys = [max([r["mean_synthetic"] for r in rows if r["layer"] == lay] or [np.nan]) for lay in layers]
    ax.bar(layers, ys, color="steelblue")
    ax.axhline(CLASSIC_BASELINE, color="crimson", ls="--", lw=2)
    ax.set_title("Глубина слоя (при лучшем входе/пулинге)")
    ax.set_ylabel("средний AUROC")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Можно ли настройкой DINOv2 побить 11 чисел классической статистики")
    fig.tight_layout()
    fig.savefig(OUT / "knobs_sweep.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
