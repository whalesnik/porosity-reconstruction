"""
Эксперимент 1 (переписан после exp00) — препроцессинг и разделимость Air/Xe.

Что изменилось против первой версии и почему:
1. Первая версия давала probe_AUC = 1.000 во ВСЕХ 12 конфигурациях. Это не
   результат, а артефакт: при 768 признаках на 80 образцов линейный классификатор
   разделяет любые две группы, даже случайные. Добавлен отрицательный контроль
   (та же метрика на случайном делении Air пополам) — он показывает цену деления.
2. Добавлена главная честная метрика — OOD-AUROC: по банку референсных Air-срезов
   считаем kNN-скор для отложенных Air и для Xe и смотрим, отделяются ли они.
   Размерность тут не мешает, потому что ничего не обучается.
3. Маска керна убрана: exp00 показал, что фона в кадре нет вообще (данные —
   уже вырезанный внутренний субобъём), поэтому кроп берётся просто из центра.
"""

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib_dino as L

N_AIR = 40
N_XE = 20
N_REF = 25          # банк "нормы" из Air
OUT = L.RUNS_DIR / "exp01_preprocessing"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
    rng = np.random.default_rng(0)

    print("Загружаю срезы...")
    air_vol, _ = L.load_air(N_AIR)
    xe_vol, _ = L.load_xe(N_XE)
    raw_stats = {"air_mean_raw": float(air_vol.mean()), "xe_mean_raw": float(xe_vol.mean())}
    print(f"  Сырая яркость: Air {raw_stats['air_mean_raw']:.1f} vs "
          f"Xe {raw_stats['xe_mean_raw']:.1f} "
          f"({raw_stats['xe_mean_raw'] - raw_stats['air_mean_raw']:+.1f})")

    g_lo, g_hi = L.compute_global_window([air_vol, xe_vol])
    crop_box = L.center_box(air_vol[0].shape)
    perm = rng.permutation(N_AIR)
    ref_idx, test_idx = perm[:N_REF], perm[N_REF:]

    backbone = L.load_dinov2("base", pooling="cls")
    results, cache = [], {}

    for norm_name, norm_fn in L.NORMALIZATIONS.items():
        for input_mode in ["resize", "crop"]:
            t0 = time.time()
            kw = {"global_lo": g_lo, "global_hi": g_hi}
            air_n = [norm_fn(v, **kw) for v in air_vol]
            xe_n = [norm_fn(v, **kw) for v in xe_vol]

            air_emb = L.embed_batch(backbone, [L.to_model_input(v, mode=input_mode, crop_box=crop_box) for v in air_n])
            xe_emb = L.embed_batch(backbone, [L.to_model_input(v, mode=input_mode, crop_box=crop_box) for v in xe_n])

            ref, air_test = air_emb[ref_idx], air_emb[test_idx]

            # Главная метрика: отличает ли kNN-детектор Xe от отложенных Air
            s_air = L.knn_score(air_test, ref, k=5, metric="cosine")
            s_xe = L.knn_score(xe_emb, ref, k=5, metric="cosine")
            ood_auroc = L.auroc(s_air, s_xe)

            # Контроль той же метрики: делим сам банк пополам — должно быть ~0.5
            h = len(ref) // 2
            ctrl_auroc = L.auroc(L.knn_score(ref[:h], ref[h:], k=5, metric="cosine"),
                                 L.knn_score(air_test, ref[h:], k=5, metric="cosine"))

            probe = L.linear_probe_auc(air_test, xe_emb)
            probe_ctrl = L.probe_control_auc(air_emb)

            mmd = L.mmd_rbf(air_test, xe_emb)
            mmd_ctrl = L.mmd_rbf(air_emb[::2], air_emb[1::2])

            key = f"{norm_name}__{input_mode}"
            cache[key] = (air_emb, xe_emb)
            row = {
                "norm": norm_name, "input": input_mode,
                "ood_auroc": ood_auroc, "ood_auroc_control": ctrl_auroc,
                "probe_auc": probe, "probe_auc_control": probe_ctrl,
                "mmd": mmd, "mmd_control": mmd_ctrl,
                "mmd_ratio": mmd / (mmd_ctrl + 1e-9),
                "mean_gap_after_norm": float(np.mean([v.mean() for v in xe_n]) - np.mean([v.mean() for v in air_n])),
                "sec": time.time() - t0,
            }
            results.append(row)
            print(f"  {key:26s} OOD-AUROC={ood_auroc:.3f} (контроль {ctrl_auroc:.3f})  "
                  f"probe={probe:.3f} (контроль {probe_ctrl:.3f})  "
                  f"MMD={mmd:.3f}/{mmd_ctrl:.3f}  ({row['sec']:.0f}s)")

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump({"raw_stats": raw_stats, "n_air": N_AIR, "n_xe": N_XE, "n_ref": N_REF,
                   "results": results}, f, indent=2, ensure_ascii=False)
    _write_table(results, raw_stats)
    _plot(results)
    _plot_tsne(cache)
    print(f"\nГотово за {time.time() - t_start:.0f}s -> {OUT}")


def _write_table(results, raw_stats):
    lines = [
        "# Эксперимент 1: препроцессинг и разделимость Air/Xe", "",
        f"Air: {N_AIR} срезов ({N_REF} в банке нормы, {N_AIR - N_REF} отложено), Xe: {N_XE}.",
        "Бэкбон: DINOv2-base, CLS-токен.", "",
        f"Сырая средняя яркость: Air={raw_stats['air_mean_raw']:.1f}, Xe={raw_stats['xe_mean_raw']:.1f} "
        f"({raw_stats['xe_mean_raw'] - raw_stats['air_mean_raw']:+.1f})", "",
        "## Метрики", "",
        "- **OOD-AUROC** — главная. kNN-расстояние до банка нормальных Air-срезов;",
        "  отличает ли оно Xe от отложенных Air. 0.5 = не отличает, 1.0 = идеально.",
        "- **контроль** рядом с каждой метрикой — то же измерение на заведомо",
        "  однородных данных (Air против Air). Показывает уровень «шума метрики».",
        "- **probe AUC** — линейный классификатор. При 768 признаках на 60 образцов",
        "  переобучается; смотреть только вместе с его контролем.",
        "- **MMD** — расстояние между распределениями, без обучения.", "",
        "| Норм. | Вход | OOD-AUROC | контроль | probe | контроль | MMD | контроль | яркостный зазор |",
        "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["ood_auroc"]):
        lines.append(f"| {r['norm']} | {r['input']} | **{r['ood_auroc']:.3f}** | {r['ood_auroc_control']:.3f} | "
                     f"{r['probe_auc']:.3f} | {r['probe_auc_control']:.3f} | "
                     f"{r['mmd']:.3f} | {r['mmd_control']:.3f} | {r['mean_gap_after_norm']:+.4f} |")
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def _plot(results):
    norms = list(L.NORMALIZATIONS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    specs = [("ood_auroc", "ood_auroc_control", "OOD-AUROC (Xe против отложенных Air)"),
             ("probe_auc", "probe_auc_control", "AUC линейного пробы"),
             ("mmd", "mmd_control", "MMD между распределениями")]
    for ax, (m, mc, title) in zip(axes, specs):
        for mode, c in [("resize", "tab:blue"), ("crop", "tab:orange")]:
            v = [next(r[m] for r in results if r["norm"] == n and r["input"] == mode) for n in norms]
            vc = [next(r[mc] for r in results if r["norm"] == n and r["input"] == mode) for n in norms]
            ax.plot(norms, v, "o-", color=c, label=f"{mode}: сигнал")
            ax.plot(norms, vc, "x--", color=c, alpha=0.5, label=f"{mode}: контроль")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Сигнал против контроля: без контроля любая метрика выглядит убедительно")
    fig.tight_layout()
    fig.savefig(OUT / "preprocessing_grid.png", dpi=150)


def _plot_tsne(cache):
    from sklearn.manifold import TSNE
    keys = ["minmax__resize", "global_window__resize", "clahe__resize", "clahe__crop"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    for ax, key in zip(axes, keys):
        air, xe = cache[key]
        X = torch.cat([air, xe]).numpy()
        Y = TSNE(n_components=2, perplexity=min(10, len(X) // 4), random_state=0).fit_transform(X)
        n = len(air)
        ax.scatter(Y[:n, 0], Y[:n, 1], c="tab:blue", s=18, label="air")
        ax.scatter(Y[n:, 0], Y[n:, 1], c="tab:red", s=18, label="xe")
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("t-SNE при разных препроцессингах")
    fig.tight_layout()
    fig.savefig(OUT / "tsne_variants.png", dpi=150)


if __name__ == "__main__":
    main()
