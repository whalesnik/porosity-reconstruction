"""
Эксперимент 5: селективность детектора — структура породы против помех съёмки.

Зачем этот эксперимент нужен.
exp02 мерил "заметить любое отличие от эталона". Для нашего проекта эта задача
поставлена неверно. Детектор доверия должен различать два случая, требующих
РАЗНЫХ действий команды:

  помехи съёмки (яркость, контраст, шум прибора) — та же порода, другой сканер.
      Правильная реакция: перенормировать данные. Флагать как "незнакомая порода" НЕЛЬЗЯ.
  структурный сдвиг (другой размер пор, другая геометрия) — другая порода.
      Правильная реакция: не доверять реконструкции, нужна доменная адаптация.

Признак, чувствительный ко всему сразу, набирает высокий AUROC в exp02 и при этом
бесполезен: он не отличает смену прибора от смены месторождения.

Метрика: селективность = AUROC(структура) - AUROC(помехи).

Отдельно считаем block_shuffle — он сохраняет гистограмму ТОЧНО (до 4-го знака),
поэтому признаки на основе распределения яркостей на нём слепы по построению.

Важное отличие от прошлых экспериментов: после exp04 известно, что настройки
DINOv2 по умолчанию (последний слой + сжатие 900->224) калечат модель. Поэтому
здесь сравниваются ОБА варианта — дефолтный и настроенный (слой 3 + нативные тайлы),
иначе выводы повторили бы ошибку exp02. Все искажения применяются к полному срезу
900x900 ДО подготовки входа, иначе тайлы в нативном разрешении не имели бы смысла.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import torch

import lib_dino as L

N_REF, N_TEST = 40, 20
SEVERITIES = [1, 2, 3, 4, 5]
OUT = L.RUNS_DIR / "exp05_structure_vs_nuisance"
PROGRESS = OUT / "progress.log"


def log(msg: str):
    """Пишем и в консоль, и в файл — чтобы за прогоном можно было следить в реальном
    времени, открыв progress.log (без буферизации через пайпы)."""
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# --------------------------------------------------------- варианты входа ---
# resize224 — то, что использовалось в exp02 (и проигрывало)
# tiles4x224 — победитель exp04: 4 окна в нативном разрешении, эмбеддинги усредняются

def as_tiles(vol01: np.ndarray, variant: str, idx: int) -> list[np.ndarray]:
    if variant == "resize224":
        return [L.prepare_2d(vol01, 224, "resize")]
    return L.tile_crops(vol01, tile=224, n_tiles=4, rng=np.random.default_rng(1000 + idx))


def build_extractors() -> dict:
    """Ключ — имя, значение — (функция, применима ли к тайлам поштучно)."""
    ex = {}
    bb_default = L.load_dinov2("small", pooling="cls")                # последний слой
    bb_tuned = L.load_dinov2("small", pooling="cls", layer=3)         # победитель exp04
    ex["dinov2-small[cls]@last"] = lambda imgs, b=bb_default: L.embed_batch(b, [L.to_tensor(i) for i in imgs])
    ex["dinov2-small[cls]@L3"] = lambda imgs, b=bb_tuned: L.embed_batch(b, [L.to_tensor(i) for i in imgs])
    ex["classic-stats"] = lambda imgs: torch.from_numpy(np.stack([L.features_basic_stats(i) for i in imgs]))
    ex["classic-histogram"] = lambda imgs: torch.from_numpy(np.stack([L.features_histogram(i) for i in imgs]))
    ex["classic-glcm"] = lambda imgs: torch.from_numpy(np.stack([L.features_glcm(i) for i in imgs]))
    return ex


def embed_slices(ex_fn, prepared: list[list[np.ndarray]]) -> torch.Tensor:
    """Эмбеддинг среза = среднее по его тайлам."""
    out = []
    for tiles in prepared:
        v = ex_fn(tiles).float()
        out.append(v.mean(dim=0, keepdim=True))
    return torch.cat(out, dim=0)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    rng = np.random.default_rng(0)

    log("Загружаю срезы...")
    air_all, _ = L.load_air(N_REF + N_TEST)
    g_lo, g_hi = L.compute_global_window([air_all])
    phys = np.stack([np.clip((v - g_lo) / (g_hi - g_lo), 0, 1).astype(np.float32) for v in air_all])
    perm = rng.permutation(len(phys))
    ref_raw, test_raw = phys[perm[:N_REF]], phys[perm[N_REF:]]

    log("Строю искажения на полном разрешении 900x900 (это самая долгая часть)...")
    sets = {"__ref__": ref_raw, "clean": test_raw}
    kinds = {}
    for group, table in (("nuisance", L.NUISANCE), ("structural", L.STRUCTURAL)):
        for cname, cfn in table.items():
            kinds[cname] = group
            t = time.time()
            for sev in SEVERITIES:
                need_rng = cname in ("noise", "elastic", "block_shuffle")
                kw = {"rng": np.random.default_rng(sev)} if need_rng else {}
                sets[f"{cname}_s{sev}"] = np.stack([cfn(v, sev, **kw) for v in test_raw])
            log(f"  {cname:14s} готов ({time.time()-t:.0f}s)")

    extractors = build_extractors()
    variants = ["resize224", "tiles4x224"]
    log(f"Экстракторов: {len(extractors)}, наборов: {len(sets)}, вариантов входа: {len(variants)}")

    rows = []
    for variant in variants:
        prepared = {k: [as_tiles(v, variant, i) for i, v in enumerate(vols)]
                    for k, vols in sets.items()}
        for ex_name, ex_fn in extractors.items():
            t = time.time()
            feats = {k: embed_slices(ex_fn, v) for k, v in prepared.items()}
            ref = feats["__ref__"]
            s_clean = L.knn_score(feats["clean"], ref, k=5, metric="cosine")

            per_kind = {}
            for set_name, f in feats.items():
                if set_name in ("__ref__", "clean"):
                    continue
                cname, _ = set_name.rsplit("_s", 1)
                per_kind.setdefault(cname, []).append(
                    L.auroc(s_clean, L.knn_score(f, ref, k=5, metric="cosine")))

            means = {k: float(np.mean(v)) for k, v in per_kind.items()}
            struct = float(np.mean([v for k, v in means.items() if kinds[k] == "structural"]))
            nuis = float(np.mean([v for k, v in means.items() if kinds[k] == "nuisance"]))
            rows.append({"variant": variant, "extractor": ex_name, "per_kind": means,
                         "structural": struct, "nuisance": nuis, "selectivity": struct - nuis,
                         "block_shuffle": means["block_shuffle"]})
            log(f"  {variant:11s} {ex_name:24s} структура={struct:.3f} помехи={nuis:.3f} "
                f"селективность={struct-nuis:+.3f} shuffle={means['block_shuffle']:.3f} ({time.time()-t:.0f}s)")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(rows, kinds, time.time() - t0)
    plot(rows, kinds)
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def write_report(rows, kinds, elapsed):
    lines = ["# Эксперимент 5: селективность — структура породы против помех съёмки", "",
             "Детектор доверия должен ловить смену породы и игнорировать смену сканера.",
             "**Селективность = AUROC(структура) − AUROC(помехи)**.", "",
             "`block_shuffle` сохраняет гистограмму точно — признаки на основе распределения",
             "яркостей обязаны показать на нём ≈0.5.", "",
             "| Вход | Экстрактор | Структура ↑ | Помехи ↓ | Селективность | block_shuffle |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["selectivity"]):
        lines.append(f"| {r['variant']} | {r['extractor']} | {r['structural']:.3f} | {r['nuisance']:.3f} | "
                     f"**{r['selectivity']:+.3f}** | {r['block_shuffle']:.3f} |")

    lines += ["", "## Разбивка по типам сдвига", "",
              "| Вход | Экстрактор | " + " | ".join(kinds) + " |",
              "|---|---|" + "---|" * len(kinds)]
    for r in sorted(rows, key=lambda r: -r["selectivity"]):
        cells = " | ".join(f"{r['per_kind'].get(k, float('nan')):.3f}" for k in kinds)
        lines.append(f"| {r['variant']} | {r['extractor']} | {cells} |")

    best = max(rows, key=lambda r: r["selectivity"])
    lines += ["", "## Итог", "",
              f"- Лучшая селективность: **{best['extractor']}** на входе `{best['variant']}` "
              f"= {best['selectivity']:+.3f} (структура {best['structural']:.3f}, помехи {best['nuisance']:.3f})",
              "", f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(rows, kinds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, variant in zip(axes, ["resize224", "tiles4x224"]):
        rs = sorted([r for r in rows if r["variant"] == variant], key=lambda r: -r["selectivity"])
        y = np.arange(len(rs))
        ax.barh(y - 0.2, [r["structural"] for r in rs], 0.4, color="seagreen",
                label="структура: другая порода (ловить)")
        ax.barh(y + 0.2, [r["nuisance"] for r in rs], 0.4, color="indianred",
                label="помехи: другой сканер (игнорировать)")
        ax.set_yticks(y)
        ax.set_yticklabels([r["extractor"] for r in rs], fontsize=8)
        ax.axvline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlim(0.3, 1.02)
        ax.invert_yaxis()
        ax.set_xlabel("AUROC")
        ax.set_title(f"вход: {variant}")
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Полезный детектор: зелёная полоса длинная, красная короткая")
    fig.tight_layout()
    fig.savefig(OUT / "selectivity.png", dpi=140, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(7.5, 4))
    rs = sorted(rows, key=lambda r: -r["block_shuffle"])
    labels = [f"{r['extractor']}\n({r['variant']})" for r in rs]
    ax.barh(labels, [r["block_shuffle"] for r in rs], color="steelblue")
    ax.axvline(0.5, color="crimson", ls="--", lw=2, label="слепота: гистограмма не изменилась")
    ax.set_xlabel("AUROC на block_shuffle")
    ax.set_title("Кто видит структуру, когда гистограмма сохранена точно")
    ax.tick_params(labelsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig2.tight_layout()
    fig2.savefig(OUT / "block_shuffle_blindness.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
