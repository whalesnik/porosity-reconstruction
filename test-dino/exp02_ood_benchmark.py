"""
Эксперимент 2 — настоящий OOD-бенчмарк: какой детектор реально ловит домен-сдвиг.

Проблема: "своего" второго датасета сегодня нет, поэтому проверить OOD-детектор
не на чем. Решение — имитировать сдвиг контролируемыми искажениями известной силы
(яркость, контраст, шум, размытие, ухудшение разрешения; 5 уровней каждое).
Это даёт то, чего нет у голого t-SNE: числовую метрику качества (AUROC) и
возможность честно сравнить методы между собой.

Важная деталь протокола: искажения применяются в ОБЩЕЙ ФИЗИЧЕСКОЙ шкале яркости
(как если бы это реально был другой сканер), а нормализация пайплайна работает
уже ПОСЛЕ них. Поэтому видно ключевой компромисс: нормализация, которая защищает
модель от сдвига яркости, одновременно делает этот сдвиг невидимым для детектора.

Сравниваются: DINOv2 (разные размеры и пулинги), ResNet50-ImageNet и классические
признаки (гистограмма, GLCM/Haralick, простая статистика) — последние нужны,
чтобы понять, даёт ли тяжёлая нейросеть что-то сверх дешёвой классики.
"""

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib_dino as L

N_REF = 40          # in-distribution "то, что модель видела"
N_TEST = 20         # чистый held-out тест (не участвует в reference)
N_XE = 20           # Xe-срезы как естественный (не синтетический) сдвиг
SEVERITIES = [1, 2, 3, 4, 5]
INPUT_MODE = "resize"
OUT = L.RUNS_DIR / "exp02_ood_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

# Нормализации пайплайна, применяемые ПОСЛЕ искажения.
PIPELINE_NORMS = ["identity", "percentile", "clahe"]


def build_extractors():
    """Каждый — функция: список (S,S) изображений в [0,1] -> тензор признаков (N,D)."""
    ex = {}

    for size, pooling in [("small", "cls"), ("base", "cls"), ("base", "mean")]:
        bb = L.load_dinov2(size, pooling=pooling)

        def make(bb=bb):
            def f(imgs):
                return L.embed_batch(bb, [L.to_tensor(im) for im in imgs])
            return f
        ex[bb.name] = make()

    rn = L.load_resnet50()

    def f_rn(imgs, rn=rn):
        return L.embed_batch(rn, [L.to_tensor(im) for im in imgs])
    ex[rn.name] = f_rn

    ex["classic-histogram"] = lambda imgs: torch.from_numpy(
        np.stack([L.features_histogram(im) for im in imgs]))
    ex["classic-glcm"] = lambda imgs: torch.from_numpy(
        np.stack([L.features_glcm(im) for im in imgs]))
    ex["classic-stats"] = lambda imgs: torch.from_numpy(
        np.stack([L.features_basic_stats(im) for im in imgs]))
    return ex


def main():
    t_start = time.time()
    rng = np.random.default_rng(0)

    print("Загружаю срезы...")
    air_all, _ = L.load_air(N_REF + N_TEST)
    xe_all, _ = L.load_xe(N_XE)
    g_lo, g_hi = L.compute_global_window([air_all])

    def to_phys(v):
        return np.clip((v - g_lo) / (g_hi - g_lo), 0, 1).astype(np.float32)

    phys = np.stack([to_phys(v) for v in air_all])
    perm = rng.permutation(len(phys))
    ref_raw, test_raw = phys[perm[:N_REF]], phys[perm[N_REF:]]
    xe_phys = np.stack([to_phys(v) for v in xe_all])
    print(f"  reference={len(ref_raw)}, clean-test={len(test_raw)}, xe={len(xe_phys)}")

    crop_box = L.center_box(air_all[0].shape)

    # --- строим все наборы изображений один раз, до извлечения признаков ---
    print("Готовлю искажённые наборы...")
    sets = {"__ref__": ref_raw, "clean": test_raw, "xe_natural": xe_phys}
    for cname, cfn in L.CORRUPTIONS.items():
        for sev in SEVERITIES:
            kw = {"rng": np.random.default_rng(sev)} if cname == "noise" else {}
            sets[f"{cname}_s{sev}"] = np.stack([cfn(v, sev, **kw) for v in test_raw])

    extractors = build_extractors()
    print(f"Экстракторов: {len(extractors)}, наборов: {len(sets)}, "
          f"нормализаций: {len(PIPELINE_NORMS)}")

    rows = []
    for norm_name in PIPELINE_NORMS:
        norm_fn = L.NORMALIZATIONS[norm_name] if norm_name != "identity" else L.norm_identity
        prepared = {k: [L.prepare_2d(norm_fn(v, global_lo=0.0, global_hi=1.0),
                                     mode=INPUT_MODE, crop_box=crop_box) for v in vols]
                    for k, vols in sets.items()}

        for ex_name, ex_fn in extractors.items():
            t0 = time.time()
            feats = {k: ex_fn(v).float() for k, v in prepared.items()}
            ref = feats["__ref__"]

            for score_name, score_fn in [
                ("knn5_cos", lambda q: L.knn_score(q, ref, k=5, metric="cosine")),
                ("mahalanobis", lambda q: L.mahalanobis_score(q, ref)),
            ]:
                s_clean = score_fn(feats["clean"])
                for set_name, f in feats.items():
                    if set_name in ("__ref__", "clean"):
                        continue
                    s_out = score_fn(f)
                    rows.append({
                        "norm": norm_name, "extractor": ex_name, "score": score_name,
                        "test_set": set_name,
                        "corruption": set_name.rsplit("_s", 1)[0] if "_s" in set_name else set_name,
                        "severity": int(set_name.rsplit("_s", 1)[1]) if "_s" in set_name else 0,
                        "auroc": L.auroc(s_clean, s_out),
                        "mean_score_clean": float(s_clean.mean()),
                        "mean_score_out": float(s_out.mean()),
                    })
            print(f"  {norm_name:11s} {ex_name:22s} ({time.time() - t0:5.1f}s)  "
                  f"dim={ref.shape[1]}")

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    _report(rows)
    print(f"\nГотово за {time.time() - t_start:.0f}s -> {OUT}")


def _report(rows):
    import collections
    norms = PIPELINE_NORMS
    extractors = sorted({r["extractor"] for r in rows})
    corrs = [c for c in L.CORRUPTIONS] + ["xe_natural"]

    lines = ["# Эксперимент 2: OOD-бенчмарк на контролируемых искажениях", "",
             "AUROC — насколько уверенно детектор отличает искажённые срезы от чистых.",
             "0.5 = не отличает вообще, 1.0 = идеально. Скор — kNN(k=5, cosine).", "",
             "Средний AUROC по всем уровням искажения (severity 1-5):", ""]

    header = "| Нормализация | Экстрактор | " + " | ".join(corrs) + " | СРЕДНЕЕ |"
    lines += [header, "|" + "---|" * (len(corrs) + 3)]

    best = []
    for norm in norms:
        for ex in extractors:
            cells, all_v = [], []
            for c in corrs:
                vals = [r["auroc"] for r in rows if r["norm"] == norm and r["extractor"] == ex
                        and r["score"] == "knn5_cos" and r["corruption"] == c]
                if vals:
                    m = float(np.mean(vals))
                    cells.append(f"{m:.3f}")
                    if c != "xe_natural":
                        all_v.append(m)
                else:
                    cells.append("—")
            avg = float(np.mean(all_v)) if all_v else 0.0
            best.append((avg, norm, ex))
            lines.append(f"| {norm} | {ex} | " + " | ".join(cells) + f" | **{avg:.3f}** |")

    best.sort(reverse=True)
    lines += ["", "## Лучшие конфигурации (по среднему AUROC на синтетических искажениях)", ""]
    for avg, norm, ex in best[:5]:
        lines.append(f"- **{avg:.3f}** — {ex} + нормализация `{norm}`")

    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")

    # --- тепловая карта: экстрактор x искажение, для каждой нормализации ---
    fig, axes = plt.subplots(1, len(norms), figsize=(6 * len(norms), 0.45 * len(extractors) + 2.5))
    for ax, norm in zip(np.atleast_1d(axes), norms):
        M = np.full((len(extractors), len(corrs)), np.nan)
        for i, ex in enumerate(extractors):
            for j, c in enumerate(corrs):
                vals = [r["auroc"] for r in rows if r["norm"] == norm and r["extractor"] == ex
                        and r["score"] == "knn5_cos" and r["corruption"] == c]
                if vals:
                    M[i, j] = np.mean(vals)
        im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(corrs)), corrs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(extractors)), extractors, fontsize=8)
        ax.set_title(f"norm = {norm}", fontsize=10)
        for i in range(len(extractors)):
            for j in range(len(corrs)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="w" if M[i, j] < 0.85 else "k")
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label="AUROC", shrink=0.8)
    fig.suptitle("Качество OOD-детекции (AUROC, среднее по severity 1-5)")
    fig.savefig(OUT / "heatmap_auroc.png", dpi=150, bbox_inches="tight")

    # --- кривые: AUROC vs severity для лучших экстракторов ---
    top = [b for b in best[:4]]
    fig2, axes2 = plt.subplots(1, len(L.CORRUPTIONS), figsize=(3.2 * len(L.CORRUPTIONS), 3.4),
                               sharey=True)
    for ax, c in zip(axes2, L.CORRUPTIONS):
        for _, norm, ex in top:
            ys = [np.mean([r["auroc"] for r in rows if r["norm"] == norm and r["extractor"] == ex
                           and r["score"] == "knn5_cos" and r["corruption"] == c
                           and r["severity"] == s]) for s in SEVERITIES]
            ax.plot(SEVERITIES, ys, "o-", label=f"{ex}/{norm}", ms=4)
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("severity")
        ax.grid(alpha=0.3)
    axes2[0].set_ylabel("AUROC")
    axes2[-1].legend(fontsize=6.5, loc="lower right")
    fig2.suptitle("Чувствительность к силе сдвига")
    fig2.tight_layout()
    fig2.savefig(OUT / "auroc_vs_severity.png", dpi=150)


if __name__ == "__main__":
    main()
