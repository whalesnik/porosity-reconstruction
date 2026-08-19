"""
Эксперимент 7: НАСТОЯЩАЯ проверка детектора на чужом датасете.

До этого момента все "чужие" данные в проекте были синтетическими (яркость, шум,
морфология пор). Здесь впервые появился реальный второй датасет: образец 22322-24,
сухой скан, 2775x2774x2420, воксель 1.84 мкм — другой керн, другой прибор,
другая съёмка. Это тот самый тест, ради которого ветка вообще существует.

Честное сравнение требует убрать два тривиальных различия, иначе детектор
сработает на них, а не на породе:

  1. Кадр. У Сколтеха фона нет (уже вырезанный субобъём 900x900), у нас есть
     чёрный фон и держатель. Поэтому свой срез сначала обрезается по вписанному
     в керн квадрату, затем берётся центральные 900x900 — тот же кадр, что у
     Сколтеха, только порода.
  2. Масштаб. У Сколтеха 3.0 мкм/воксель, у нас 1.84 мкм/воксель — то есть наш
     скан МЕЛЬЧЕ, и его надо огрублять. Без поправки "другая порода" смешается
     с "другое увеличение".

     ИСПРАВЛЕНО (см. exp12). Раньше здесь стояло 1.2 мкм для Сколтеха — число
     из смежной статьи, к этому датасету не относящееся. Проверка по данным
     команды: срезы ood_testset для Сколтеха это побитово точные центральные
     кропы наших PNG, а в labels.csv у них указано 3.0 мкм. Заодно в формуле
     ниже отношение стояло перевёрнутым, и две ошибки почти скомпенсировали
     друг друга: пересчёт давал 2.82 мкм вместо нужных 3.0, промах 6%. Поэтому
     вывод эксперимента устоял, но обоснование было неверным.

Три варианта подачи своих данных, каждый отвечает на свой вопрос:

  as_is         — окно яркости от Сколтеха, масштаб свой.
                  Вопрос: что увидит наивный пользователь, подавший чужие данные.
  renormalized  — окно яркости своё (перцентильное), масштаб свой.
                  Вопрос: остаётся ли сдвиг после честной перенормировки.
  rescaled      — окно своё + пересчёт 1.84 -> 3.0 мкм/воксель.
                  Вопрос: остаётся ли сдвиг, когда и яркость, и масштаб выровнены.
                  Если да — это уже настоящее различие пород.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import lib_dino as L
from domain_guard import DomainGuard

OWN_DIR = L.PROJECT_ROOT / "22322-24_dry_4mm_1.8um_16bit"
OUT = L.RUNS_DIR / "exp07_real_domain_shift"
N_REF, N_CTRL = 40, 20
FRAME = 900                    # общий размер кадра для обоих датасетов
VOX_OWN, VOX_SK = 1.84, 3.0    # мкм на воксель (Сколтех — из labels.csv команды)
PROGRESS = OUT / "progress.log"


def log(msg: str):
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_own_frames(rescale: bool) -> np.ndarray:
    """Свои срезы, обрезанные до кадра 900x900 чистой породы.

    rescale=True — берём область в VOX_SK/VOX_OWN раз больше и сжимаем до 900,
    чтобы после сжатия пиксель нёс те же VOX_SK мкм, что у Сколтеха. Наш скан
    мельче (1.84 против 3.0), поэтому его надо огрублять: берём 1467 пикселей
    и ужимаем в 900."""
    files = sorted(OWN_DIR.glob("rec_*.tif"), key=lambda p: int(p.stem.split("_")[-1]))
    want = int(round(FRAME * VOX_SK / VOX_OWN)) if rescale else FRAME
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


def to01(vols: np.ndarray, window: tuple[float, float]) -> list[np.ndarray]:
    lo, hi = window
    return [np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float32) for v in vols]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")

    log("Загружаю Сколтех (эталон)...")
    sk_all, _ = L.load_air(N_REF + N_CTRL)
    # ВАЖНО: делить случайно, а не последовательно. load_air отдаёт срезы
    # равномерно по глубине, поэтому sk_all[:40] это верхние две трети керна,
    # а sk_all[40:] — нижняя треть. При последовательном делении контроль
    # оказывается систематически с другой глубины, чем эталон, и часть «ложных
    # тревог» объясняется не детектором, а тем, что низ образца просто другой.
    sk_perm = np.random.default_rng(0).permutation(len(sk_all))
    sk_ref, sk_ctrl = sk_all[sk_perm[:N_REF]], sk_all[sk_perm[N_REF:]]

    log("Загружаю свой датасет (49 срезов, обрезка по керну)...")
    own_native = load_own_frames(rescale=False)
    own_rescaled = load_own_frames(rescale=True)
    log(f"  свой native:   {own_native.shape}, среднее={own_native.mean():.0f}")
    log(f"  свой rescaled: {own_rescaled.shape}, среднее={own_rescaled.mean():.0f}")

    log("Калибрую guard на эталоне Сколтеха...")
    guard = DomainGuard.fit(list(sk_ref))
    sk_window = guard.window
    own_window = L.compute_global_window([own_native])
    log(f"  окно Сколтеха: {sk_window[0]:.0f}..{sk_window[1]:.0f}")
    log(f"  окно своё:     {own_window[0]:.0f}..{own_window[1]:.0f}")

    # варианты подачи: (имя, срезы, окно яркости)
    variants = {
        "as_is": (own_native, sk_window),
        "renormalized": (own_native, own_window),
        "rescaled": (own_rescaled, L.compute_global_window([own_rescaled])),
    }

    log("\nПроверяю контроль (отложенные срезы Сколтеха)...")
    ctrl = [guard.check(s) for s in sk_ctrl]
    ctrl_a = np.array([r.acquisition for r in ctrl])
    ctrl_s = np.array([r.structure for r in ctrl])
    log(f"  контроль: съёмка {ctrl_a.mean():+.2f}±{ctrl_a.std():.2f}, "
        f"структура {ctrl_s.mean():+.2f}±{ctrl_s.std():.2f}")

    rows = {"control_skoltech": {"acq": ctrl_a.tolist(), "str": ctrl_s.tolist()}}
    maps = {}
    for name, (vols, window) in variants.items():
        log(f"\nВариант '{name}'...")
        # подставляем окно варианта: guard нормализует по своему окну, поэтому
        # заранее переводим срез в шкалу Сколтеха через нужное окно
        lo, hi = window
        slo, shi = sk_window
        scaled = [np.clip((v - lo) / (hi - lo), 0, 1) * (shi - slo) + slo for v in vols]
        res = [guard.check(s) for s in scaled]
        a = np.array([r.acquisition for r in res])
        s = np.array([r.structure for r in res])
        rows[name] = {"acq": a.tolist(), "str": s.tolist(),
                      "auroc_acq": L.auroc(torch.tensor(ctrl_a), torch.tensor(a)),
                      "auroc_str": L.auroc(torch.tensor(ctrl_s), torch.tensor(s)),
                      "verdicts": [r.verdict for r in res]}
        maps[name] = res[0].structure_map
        log(f"  съёмка:    {a.mean():+8.1f} ± {a.std():6.1f}   AUROC={rows[name]['auroc_acq']:.3f}")
        log(f"  структура: {s.mean():+8.1f} ± {s.std():6.1f}   AUROC={rows[name]['auroc_str']:.3f}")
        n_alarm = sum("СТОП" in v for v in rows[name]["verdicts"])
        log(f"  вердикт 'СТОП: структура незнакомая' у {n_alarm}/{len(res)} срезов")

    maps["control"] = ctrl[0].structure_map
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(rows, maps, ctrl_a, ctrl_s)
    write_report(rows, sk_window, own_window, time.time() - t0)
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


def plot(rows, maps, ctrl_a, ctrl_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["as_is", "renormalized", "rescaled"]
    colors = {"as_is": "tab:red", "renormalized": "tab:orange", "rescaled": "tab:green"}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(ctrl_a, ctrl_s, s=55, c="tab:blue", label="Сколтех (контроль)", zorder=3)
    for n in names:
        ax.scatter(rows[n]["acq"], rows[n]["str"], s=45, c=colors[n], alpha=0.75,
                   label=f"свой: {n}", zorder=3)
    ax.axhline(6, color="crimson", ls="--", lw=1.4)
    ax.axvline(6, color="crimson", ls="--", lw=1.4)
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("канал съёмки (z): сменился прибор →")
    ax.set_ylabel("канал структуры (z): сменилась порода →")
    ax.set_title("Реальный домен-сдвиг: свой датасет против эталона Сколтеха")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "real_shift_two_channels.png", dpi=140, bbox_inches="tight")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (key, title) in enumerate([("str", "канал структуры"), ("acq", "канал съёмки")]):
        ax = axes[i]
        data = [ctrl_s if key == "str" else ctrl_a] + [np.array(rows[n][key]) for n in names]
        ax.boxplot(data, tick_labels=["контроль\nСколтех"] + names, showfliers=False)
        ax.axhline(6, color="crimson", ls="--", lw=1.4, label="порог тревоги")
        ax.set_yscale("symlog")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig2.suptitle("Что происходит с каждым каналом по мере выравнивания условий")
    fig2.tight_layout()
    fig2.savefig(OUT / "channels_by_variant.png", dpi=140, bbox_inches="tight")

    fig3, axes = plt.subplots(1, len(maps), figsize=(3.2 * len(maps), 3.6))
    vmax = max(m.max() for m in maps.values())
    for ax, (name, m) in zip(axes, maps.items()):
        im = ax.imshow(m, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig3.colorbar(im, ax=axes.tolist(), shrink=0.8, label="структурная непохожесть")
    fig3.suptitle("Карта структурного канала на реальных чужих данных")
    fig3.savefig(OUT / "structure_maps_real.png", dpi=140, bbox_inches="tight")


def write_report(rows, sk_window, own_window, elapsed):
    ca, cs = np.array(rows["control_skoltech"]["acq"]), np.array(rows["control_skoltech"]["str"])
    lines = ["# Эксперимент 7: реальный домен-сдвиг (свой датасет 22322-24)", "",
             "Первая проверка детектора на настоящих чужих данных, а не на синтетике.", "",
             f"- окно яркости Сколтеха: {sk_window[0]:.0f}..{sk_window[1]:.0f}",
             f"- окно яркости своё: {own_window[0]:.0f}..{own_window[1]:.0f}",
             f"- масштаб: свой {VOX_OWN} мкм/вокс против {VOX_SK} мкм/вокс у Сколтеха "
             f"(отношение {VOX_OWN/VOX_SK:.2f})", "",
             "| Набор | канал съёмки (z) | канал структуры (z) | AUROC съёмка | AUROC структура | «СТОП» |",
             "|---|---|---|---|---|---|",
             f"| контроль Сколтех | {ca.mean():+.2f} ± {ca.std():.2f} | {cs.mean():+.2f} ± {cs.std():.2f} | — | — | — |"]
    for n in ["as_is", "renormalized", "rescaled"]:
        a, s = np.array(rows[n]["acq"]), np.array(rows[n]["str"])
        n_alarm = sum("СТОП" in v for v in rows[n]["verdicts"])
        lines.append(f"| свой: {n} | {a.mean():+.1f} ± {a.std():.1f} | {s.mean():+.1f} ± {s.std():.1f} | "
                     f"{rows[n]['auroc_acq']:.3f} | {rows[n]['auroc_str']:.3f} | {n_alarm}/{len(a)} |")
    lines += ["", "## Как читать", "",
              "- `as_is` — что увидит человек, подавший чужие данные без подготовки.",
              "- `renormalized` — убрана разница в яркости, остался масштаб и порода.",
              "- `rescaled` — убраны и яркость, и масштаб. Остаток — различие самих пород.", "",
              f"_Время: {elapsed:.0f}s_"]
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
