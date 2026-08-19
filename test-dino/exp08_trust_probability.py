"""
Эксперимент 8: превращаем z-оценку в число 0..1 — «насколько данные знакомы».

Задача от команды: по новому срезу выдавать вероятность, с которой можно доверять
предсказанию реконструкции на этом срезе.

ВАЖНОЕ УТОЧНЕНИЕ ФОРМУЛИРОВКИ. Прямо сейчас честно считается вероятность того,
что срез ПОХОЖ на данные, на которых модель обучалась. Это не то же самое, что
вероятность правильности предсказания: чтобы утверждать второе, нужно показать,
что ошибка реконструкции действительно растёт вместе с этим числом. Модели
реконструкции пока нет, значит связь не измерена. Проверяется одним экспериментом,
когда она появится (см. блок «что проверить дальше» в конце отчёта).

Считаются две величины, обе на структурном канале:

  trust      — 0..1, гауссов хвост по калибровочному разбросу эталона.
               Типичный срез = 1.0, дальше монотонно падает. Удобно для показа,
               но опирается на предположение о нормальности хвоста.
  p_typical  — непараметрический p-value: доля эталонных срезов, оказавшихся
               не ближе к норме, чем проверяемый. Ничего не предполагает,
               но разрешение ограничено размером эталона (при 40 срезах ~0.024).

Проверка на трёх наборах:
  - отложенные срезы Сколтеха (эталонный домен)  -> ожидаем высокое доверие;
  - свой датасет 22322-24, три варианта подготовки -> ожидаем низкое,
    причём растущее по мере выравнивания яркости и масштаба (как в exp07).
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import lib_dino as L
from domain_guard import DomainGuard

OWN_DIR = L.PROJECT_ROOT / "22322-24_dry_4mm_1.8um_16bit"
OUT = L.RUNS_DIR / "exp08_trust_probability"
N_REF, N_CTRL = 40, 20
FRAME = 900
VOX_OWN, VOX_SK = 1.84, 1.2
PROGRESS = OUT / "progress.log"


def log(msg: str):
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_own_frames(rescale: bool) -> np.ndarray:
    """То же, что в exp07: обрезка по керну до кадра 900x900 чистой породы."""
    files = sorted(OWN_DIR.glob("rec_*.tif"), key=lambda p: int(p.stem.split("_")[-1]))
    want = int(round(FRAME * VOX_OWN / VOX_SK)) if rescale else FRAME
    out = []
    for f in files:
        a = np.array(Image.open(f)).astype(np.float32)
        r0, c0, size = L.inscribed_box_from_mask(L.core_mask_from_slice(a))
        cy, cx = r0 + size // 2, c0 + size // 2
        half = want // 2
        crop = a[cy - half:cy + half, cx - half:cx + half]
        if rescale:
            t = torch.from_numpy(crop)[None, None]
            crop = F.interpolate(t, size=(FRAME, FRAME), mode="area")[0, 0].numpy()
        out.append(crop.astype(np.float32))
    return np.stack(out)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")

    log("Строю банк на эталоне Сколтеха (40 сухих срезов)...")
    sk_all, _ = L.load_air(N_REF + N_CTRL)
    # делим случайно, а не последовательно: load_air отдаёт срезы равномерно по
    # глубине, и sk_all[:40] это верхние две трети керна, а sk_all[40:] — нижняя
    # треть. При последовательном делении контроль систематически с другой
    # глубины, чем эталон, и это подмешивается к «ложным тревогам».
    sk_perm = np.random.default_rng(0).permutation(len(sk_all))
    sk_ref, sk_ctrl = sk_all[sk_perm[:N_REF]], sk_all[sk_perm[N_REF:]]
    guard = DomainGuard.fit(list(sk_ref))
    sk_window = guard.window
    log(f"  банк структуры: {tuple(guard.ref_tokens.shape)}")
    log(f"  калибровочный разброс: mu={guard.mu_s:.4f} sd={guard.sd_s:.4f}")

    results = {}

    log("\nКонтроль — отложенные срезы Сколтеха (в банк не входили):")
    ctrl = [guard.check(s) for s in sk_ctrl]
    results["Сколтех (отложенные)"] = ctrl
    for r in ctrl[:5]:
        log(f"  доверие={r.trust:.3f}  p={r.p_typical:.3f}  z={r.structure:+.2f}")
    log(f"  ИТОГО: доверие {np.mean([r.trust for r in ctrl]):.3f} "
        f"(медиана {np.median([r.trust for r in ctrl]):.3f}), "
        f"выше 0.05 у {sum(r.trust > 0.05 for r in ctrl)}/{len(ctrl)}")

    log("\nСвой датасет 22322-24 (в банк не входил):")
    own_native = load_own_frames(rescale=False)
    own_rescaled = load_own_frames(rescale=True)
    own_window = L.compute_global_window([own_native])
    variants = {
        "свой: как есть": (own_native, sk_window),
        "свой: выровнена яркость": (own_native, own_window),
        "свой: яркость + масштаб": (own_rescaled, L.compute_global_window([own_rescaled])),
    }
    for name, (vols, window) in variants.items():
        lo, hi = window
        slo, shi = sk_window
        scaled = [np.clip((v - lo) / (hi - lo), 0, 1) * (shi - slo) + slo for v in vols]
        res = [guard.check(s) for s in scaled]
        results[name] = res
        log(f"  {name:26s} доверие={np.mean([r.trust for r in res]):.4f} "
            f"p={np.mean([r.p_typical for r in res]):.4f} "
            f"z={np.mean([r.structure for r in res]):+.1f}  "
            f"выше 0.05 у {sum(r.trust > 0.05 for r in res)}/{len(res)}")

    save(results, guard, time.time() - t0)
    plot(results)
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def save(results, guard, elapsed):
    payload = {name: {"trust": [r.trust for r in rs],
                      "p_typical": [r.p_typical for r in rs],
                      "z_structure": [r.structure for r in rs],
                      "z_acquisition": [r.acquisition for r in rs]}
               for name, rs in results.items()}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    lines = ["# Эксперимент 8: вероятность доверия к срезу", "",
             "Банк построен на 40 сухих срезах Сколтеха. Ни один проверяемый срез в банк не входил.", "",
             "| Набор | доверие (среднее) | доверие (медиана) | p_typical | z структуры | доверие > 0.05 |",
             "|---|---|---|---|---|---|"]
    for name, rs in results.items():
        t = np.array([r.trust for r in rs])
        p = np.array([r.p_typical for r in rs])
        z = np.array([r.structure for r in rs])
        lines.append(f"| {name} | {t.mean():.3f} | {np.median(t):.3f} | {p.mean():.3f} | "
                     f"{z.mean():+.2f} | {int((t > 0.05).sum())}/{len(t)} |")

    lines += ["", "## Как читать эти числа", "",
              "- **доверие** — 0..1, «насколько срез знаком банку». Типичный эталонный срез даёт 1.0.",
              "  Ниже ~0.02 значение не стоит читать буквально: при 40 эталонных срезах хвост",
              "  распределения экстраполирован, и малые числа означают просто «далеко за пределами известного».",
              "- **p_typical** — непараметрическая версия того же, без предположения о нормальности.",
              "  Разрешение ограничено размером банка: при 40 срезах минимальное значение ~0.024.",
              "", "## Чего это число НЕ означает", "",
              "Это вероятность того, что данные ЗНАКОМЫ модели, а не вероятность того,",
              "что предсказание реконструкции верно. Второе требует отдельной проверки.", "",
              "## Что проверить дальше, когда появится обученная модель реконструкции", "",
              "1. Прогнать модель Людей 2/3 на срезах с разным доверием.",
              "2. Посчитать реальную ошибку реконструкции (MSE/SSIM, а лучше — ошибку итоговой пористости).",
              "3. Построить график «доверие против ошибки». Если связь есть — число можно называть",
              "   доверием к предсказанию. Если нет — оно остаётся мерой знакомости данных,",
              "   что тоже полезно, но формулировку придётся менять.",
              "4. Расширить банк на весь обучающий набор основной модели — это поднимет",
              f"   разрешение шкалы (сейчас {1/(N_REF+1):.3f} из-за 40 срезов).",
              "", f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    ax = axes[0]
    data = [[r.trust for r in results[n]] for n in names]
    parts = ax.boxplot(data, tick_labels=[n.replace(": ", ":\n") for n in names],
                       showfliers=True, patch_artist=True)
    for patch, color in zip(parts["boxes"], ["tab:blue", "tab:red", "tab:orange", "tab:green"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0.05, color="crimson", ls="--", lw=1.5, label="порог 0.05")
    ax.set_yscale("log")
    ax.set_ylabel("доверие (0..1)")
    ax.set_title("Доверие к срезу: банк построен на Сколтехе")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for n, color in zip(names, ["tab:blue", "tab:red", "tab:orange", "tab:green"]):
        z = [r.structure for r in results[n]]
        t = [r.trust for r in results[n]]
        ax.scatter(z, t, s=32, alpha=0.7, c=color, label=n)
    zz = np.linspace(-3, 14, 300)
    ax.plot(zz, [DomainGuard._trust(v) for v in zz], "k-", lw=1, alpha=0.5,
            label="функция перевода z → доверие")
    ax.set_yscale("log")
    ax.set_xlabel("z структурного канала")
    ax.set_ylabel("доверие")
    ax.set_title("Как z-оценка переводится в доверие")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "trust_probability.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
