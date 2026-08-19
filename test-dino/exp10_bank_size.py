"""
Эксперимент 10: помогает ли широкий банк против ложных тревог.

Вопрос. В exp08 два знакомых среза Сколтеха из двадцати сами получили доверие
ниже порога — 10% ложных тревог на данных, которые заведомо в норме. Гипотеза
была: банк из 40 срезов слишком узок, чтобы описать естественную изменчивость.

Как проверяется. Ключ к честной постановке — зафиксировать ОБЩИЙ размер банка и
менять только то, из скольких срезов он набран:

    20 срезов  x 2000 патчей  = 40 000 векторов
    800 срезов x   50 патчей  = 40 000 векторов

Если бы мы просто брали больше срезов целиком, вырос бы и объём банка, и его
разнообразие — и нельзя было бы сказать, что именно помогло. При фиксированном
объёме единственная переменная это разнообразие.

Данные берутся из готового банка exp09 (все 900 срезов уже прогнаны через сеть),
поэтому эксперимент считает только расстояния и идёт минуты, а не часы.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import lib_dino as L

BANK_DIR = L.RUNS_DIR / "bank_full"
OUT = L.RUNS_DIR / "exp10_bank_size"
BANK_TOTAL = 40_000          # общий размер банка, одинаковый во всех условиях
N_HELDOUT = 100              # срезы Сколтеха, не входящие ни в один банк
QUERY_TOKENS = 512           # сколько патчей брать от проверяемого среза
SLICE_COUNTS = [20, 40, 100, 200, 400, 800]
CAL_SLICES = 150             # сколько срезов банка использовать для калибровки
K = 5
PROGRESS = OUT / "progress.log"


def log(msg: str):
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def trust_from_z(z: float) -> float:
    from math import erf, sqrt
    return float(min(1.0, 2.0 * (0.5 * (1.0 - erf(z / sqrt(2.0))))))


def load_bank(name: str, meta: dict) -> np.memmap:
    n = meta[f"n_{name}"]
    return np.memmap(BANK_DIR / f"{name}_tokens.f16", dtype=np.float16, mode="r",
                     shape=(n, meta["tokens_per_slice"], meta["dim"]))


def sample_tokens(bank, slice_ids: np.ndarray, per_slice: int, rng) -> torch.Tensor:
    """Стратифицированная выборка: одинаковое число патчей с каждого среза."""
    t = bank.shape[1]
    out = []
    for i in slice_ids:
        idx = rng.choice(t, size=min(per_slice, t), replace=False)
        out.append(np.asarray(bank[i][np.sort(idx)], dtype=np.float32))
    return F.normalize(torch.from_numpy(np.concatenate(out, axis=0)), dim=1)


def score_slices(queries: torch.Tensor, bank_vecs: torch.Tensor, n_q_tokens: int) -> np.ndarray:
    """Средний скор kNN для каждого среза. queries: (n_slices * n_q_tokens, D)."""
    out = []
    for i in range(0, len(queries), n_q_tokens):
        d = 1 - queries[i:i + n_q_tokens] @ bank_vecs.T
        out.append(float(d.topk(min(K, d.shape[1]), largest=False).values.mean(dim=1).mean()))
    return np.array(out)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    rng = np.random.default_rng(0)

    meta = json.loads((BANK_DIR / "meta.json").read_text(encoding="utf-8"))
    air = load_bank("air", meta)
    n_air = meta["n_air"]
    log(f"Банк: {n_air} срезов x {meta['tokens_per_slice']} патчей x {meta['dim']}")

    # разделяем срезы: часть никогда не попадает в банк и служит контролем
    perm = rng.permutation(n_air)
    heldout_ids, pool_ids = perm[:N_HELDOUT], perm[N_HELDOUT:]
    log(f"Контроль: {len(heldout_ids)} срезов, пул для банка: {len(pool_ids)}")

    # запросы готовим один раз — они не зависят от условия
    q_rng = np.random.default_rng(1)
    queries = sample_tokens(air, heldout_ids, QUERY_TOKENS, q_rng)
    log(f"Запросы: {tuple(queries.shape)}\n")

    rows = []
    for n_slices in SLICE_COUNTS:
        per_slice = min(BANK_TOTAL // n_slices, meta["tokens_per_slice"])
        ids = pool_ids[:n_slices]
        b_rng = np.random.default_rng(2)
        bank_vecs = sample_tokens(air, ids, per_slice, b_rng)

        t = time.time()
        # Калибровка: leave-one-out по срезам банка. Среднее и разброс сходятся
        # быстро, поэтому при большом банке хватает подвыборки срезов — иначе при
        # 800 срезах это семь минут счёта и гигабайт памяти на одни лишь запросы.
        cal_rng = np.random.default_rng(3)
        cal_ids = np.arange(len(ids)) if len(ids) <= CAL_SLICES else \
            cal_rng.choice(len(ids), size=CAL_SLICES, replace=False)
        cal_scores = []
        for j in cal_ids:
            q = sample_tokens(air, [ids[j]], QUERY_TOKENS, cal_rng)
            d = 1 - q @ bank_vecs.T
            lo, hi = int(j) * per_slice, (int(j) + 1) * per_slice
            d[:, lo:hi] = float("inf")          # исключаем собственные патчи
            cal_scores.append(float(d.topk(min(K, d.shape[1] - per_slice), largest=False)
                                    .values.mean(dim=1).mean()))
        cal = np.array(cal_scores)
        mu, sd = cal.mean(), cal.std() + 1e-8

        held = score_slices(queries, bank_vecs, QUERY_TOKENS)
        z = (held - mu) / sd
        trust = np.array([trust_from_z(v) for v in z])
        fa = float((trust < 0.05).mean())

        rows.append({"n_slices": int(n_slices), "per_slice": int(per_slice),
                     "bank_vectors": int(bank_vecs.shape[0]),
                     "mu": float(mu), "sd": float(sd),
                     "z_mean": float(z.mean()), "z_p95": float(np.percentile(z, 95)),
                     "trust_median": float(np.median(trust)),
                     "false_alarm_rate": fa,
                     "trust": trust.tolist(), "z": z.tolist()})
        log(f"  {n_slices:3d} срезов x {per_slice:4d} патчей = {bank_vecs.shape[0]:6d} векторов | "
            f"ложных тревог {fa*100:5.1f}% | доверие(медиана)={np.median(trust):.3f} | "
            f"z={z.mean():+.2f} | ({time.time()-t:.0f}s)")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    report(rows, time.time() - t0)
    plot(rows)
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def report(rows, elapsed):
    lines = ["# Эксперимент 10: размер банка против ложных тревог", "",
             f"Общий размер банка зафиксирован ({BANK_TOTAL} векторов) во всех условиях —",
             "меняется только число срезов, из которых он набран. Так изолируется",
             "разнообразие эталона от его объёма.", "",
             f"Контроль: {N_HELDOUT} срезов Сколтеха, не входящих ни в один банк.", "",
             "| Срезов в банке | Патчей со среза | Ложных тревог | Доверие (медиана) | z (среднее) |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['n_slices']} | {r['per_slice']} | **{r['false_alarm_rate']*100:.1f}%** | "
                     f"{r['trust_median']:.3f} | {r['z_mean']:+.2f} |")
    best, worst = min(rows, key=lambda r: r["false_alarm_rate"]), max(rows, key=lambda r: r["false_alarm_rate"])
    lines += ["", "## Итог", "",
              f"- Меньше всего ложных тревог: **{best['false_alarm_rate']*100:.1f}%** "
              f"при {best['n_slices']} срезах в банке.",
              f"- Больше всего: {worst['false_alarm_rate']*100:.1f}% при {worst['n_slices']} срезах.",
              "", f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ns = [r["n_slices"] for r in rows]

    ax = axes[0]
    ax.plot(ns, [r["false_alarm_rate"] * 100 for r in rows], "o-", lw=2, color="crimson")
    ax.set_xscale("log")
    ax.set_xlabel("срезов в банке (общий объём фиксирован)")
    ax.set_ylabel("ложных тревог, %")
    ax.set_title("Ложные тревоги на знакомых данных")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.boxplot([r["trust"] for r in rows], tick_labels=[str(n) for n in ns], showfliers=True)
    ax.axhline(0.05, color="crimson", ls="--", lw=1.4, label="порог 0.05")
    ax.set_yscale("log")
    ax.set_xlabel("срезов в банке")
    ax.set_ylabel("доверие")
    ax.set_title("Распределение доверия на контрольных срезах")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Разнообразие эталона при фиксированном объёме банка")
    fig.tight_layout()
    fig.savefig(OUT / "bank_size.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
