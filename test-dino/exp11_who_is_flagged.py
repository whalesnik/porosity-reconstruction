"""
Эксперимент 11: кто именно попадает в 5% «ложных тревог».

Из exp10: доля срабатываний на знакомых данных держится на 5-6% и НЕ падает при
росте банка с 20 до 800 срезов. Гипотеза «банк слишком мал» опровергнута.

Остаётся альтернатива: это не ложные тревоги, а настоящие — срезы, которые
действительно нетипичны для образца (трещины, крупные включения, дефекты
реконструкции, край керна). Проверяем прямо: смотрим на эти срезы глазами и
сравниваем их с самыми «нормальными» по мнению детектора.

Если у подсвеченных срезов видна общая физическая особенность — детектор прав,
и 5% это не шум метода, а неоднородность самого керна.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import lib_dino as L

OUT = L.RUNS_DIR / "exp11_who_is_flagged"
EXP10 = L.RUNS_DIR / "exp10_bank_size"
N_HELDOUT = 100
CONDITION = 800          # смотрим на условие с самым широким банком


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((EXP10 / "results.json").read_text(encoding="utf-8"))
    row = next(r for r in rows if r["n_slices"] == CONDITION)
    trust = np.array(row["trust"])
    z = np.array(row["z"])

    # те же индексы, что в exp10: rng(0).permutation(900)[:100]
    perm = np.random.default_rng(0).permutation(900)
    heldout = perm[:N_HELDOUT]
    files = L.list_slice_files(L.AIR_DIR, "Air")

    order = np.argsort(-z)                      # самые подозрительные первыми
    flagged = order[trust[order] < 0.05]
    normal = order[::-1][:4]
    print(f"Подсвечено {len(flagged)} срезов из {N_HELDOUT}")
    print("подозрительные (глубина, z):",
          [(int(heldout[i]), round(float(z[i]), 1)) for i in flagged])

    # 1) есть ли связь с глубиной в керне
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.scatter(heldout, z, s=28, c=["crimson" if t < 0.05 else "steelblue" for t in trust])
    ax.axhline(np.percentile(z, 95), color="gray", ls=":", lw=1)
    ax.set_xlabel("номер среза в керне (глубина)")
    ax.set_ylabel("z структурного канала")
    ax.set_title("Подсвеченные срезы против положения по глубине")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(z, bins=30, color="steelblue", alpha=0.8)
    for i in flagged:
        ax.axvline(z[i], color="crimson", lw=1)
    ax.set_xlabel("z структурного канала")
    ax.set_ylabel("срезов")
    ax.set_title("Распределение: хвост, а не отдельная группа?")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "depth_and_distribution.png", dpi=140, bbox_inches="tight")

    # 2) как эти срезы выглядят рядом с самыми обычными.
    # ВАЖНО: единое окно яркости на все картинки. С автоподбором на каждую
    # картинку отдельно (по умолчанию в imshow) сравнивать нельзя — тёмный срез
    # растянется на весь диапазон и будет выглядеть как светлый.
    meta = json.loads((L.RUNS_DIR / "bank_full" / "meta.json").read_text(encoding="utf-8"))
    vlo, vhi = meta["window"]
    n_show = min(4, len(flagged))
    fig2, axes = plt.subplots(2, max(n_show, 1), figsize=(3.4 * max(n_show, 1), 7.2))
    axes = np.atleast_2d(axes)
    for row_i, (group, label) in enumerate([(flagged, "подсвечен"), (normal, "самый обычный")]):
        for k in range(n_show):
            i = group[k]
            img = np.array(Image.open(files[heldout[i]])).astype(np.float32)
            axes[row_i, k].imshow(img, cmap="gray", vmin=vlo, vmax=vhi)
            axes[row_i, k].set_title(f"{label}: срез {heldout[i]}\nz={z[i]:+.1f}, "
                                     f"доверие={trust[i]:.4f}", fontsize=9)
            axes[row_i, k].axis("off")
    fig2.suptitle("Сверху — нетипичное по мнению детектора. Снизу — обычное. Единое окно яркости.")
    fig2.tight_layout()
    fig2.savefig(OUT / "flagged_vs_normal.png", dpi=130, bbox_inches="tight")

    # 3) количественная проверка догадки «дело в тяжёлых включениях»:
    # доля очень ярких вокселей (плотные минеральные зёрна) против z
    bright = []
    for i in range(N_HELDOUT):
        img = np.array(Image.open(files[heldout[i]])).astype(np.float32)
        bright.append(float((img > vhi).mean()))
    bright = np.array(bright)
    r = float(np.corrcoef(bright, z)[0, 1])
    fig3, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(bright * 100, z, s=30,
               c=["crimson" if t < 0.05 else "steelblue" for t in trust])
    ax.set_xlabel("доля ярких включений в срезе, %")
    ax.set_ylabel("z структурного канала")
    ax.set_title(f"Связь с плотными включениями: корреляция r = {r:.2f}")
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUT / "inclusions_vs_z.png", dpi=140, bbox_inches="tight")
    print(f"корреляция доли ярких включений с z: r = {r:.3f}")
    print(f"  подсвеченные: {bright[flagged].mean()*100:.3f}% ярких вокселей")
    print(f"  остальные:    {np.delete(bright, flagged).mean()*100:.3f}%")
    extra = {"corr_bright_z": r,
             "bright_flagged": float(bright[flagged].mean()),
             "bright_rest": float(np.delete(bright, flagged).mean())}

    note = ["# Кто попадает в 5% срабатываний", "",
            f"Условие: банк из {CONDITION} срезов, контроль {N_HELDOUT} срезов.", "",
            f"- Подсвечено: **{len(flagged)}** срезов",
            f"- Их номера по глубине: {sorted(int(heldout[i]) for i in flagged)}",
            f"- z у них: {[round(float(z[i]), 1) for i in flagged]}",
            f"- z у остальных: медиана {np.median(z):.2f}, 95-й перцентиль {np.percentile(z, 95):.2f}", "",
            "## Проверка догадки про плотные включения", "",
            f"- корреляция доли ярких вокселей с z: **r = {extra['corr_bright_z']:.2f}**",
            f"- у подсвеченных срезов ярких вокселей {extra['bright_flagged']*100:.3f}%",
            f"- у остальных {extra['bright_rest']*100:.3f}%", "",
            "Если подсвеченные срезы визуально отличаются (трещина, крупное включение,",
            "дефект реконструкции) — это не ошибка детектора, а реальная неоднородность",
            "керна, и называть их ложными тревогами неправильно."]
    (OUT / "note.md").write_text("\n".join(note), encoding="utf-8")
    print(f"Готово -> {OUT}")


if __name__ == "__main__":
    main()
