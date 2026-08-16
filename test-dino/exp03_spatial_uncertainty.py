"""
Эксперимент 3 — карта "где именно модель сомневается", а не одно число на срез.

Зачем: одно число на срез ("этот срез OOD") практически бесполезно для команды.
Гораздо полезнее пространственная карта: она накладывается на φ-map и показывает,
в каких именно участках предсказанию модели доверять не стоит. Это ровно то,
что нужно от ветки "модель сомневается".

Как: ViT режет изображение на патчи 14x14 и выдаёт отдельный вектор на каждый патч.
Строим банк патч-токенов по референсным (нормальным) срезам и для нового
изображения считаем для каждого патча расстояние до ближайших соседей в банке.
Получается тепловая карта аномальности в координатах изображения.
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib_dino as L

N_REF = 24
OUT = L.RUNS_DIR / "exp03_spatial_uncertainty"
OUT.mkdir(parents=True, exist_ok=True)
PATCH = 14
SIZE = 224
GRID = SIZE // PATCH  # 16x16 патчей


@torch.no_grad()
def patch_tokens(model, imgs2d: list[np.ndarray], batch_size: int = 8) -> torch.Tensor:
    """(N, GRID*GRID, D) — по вектору на каждый патч изображения."""
    outs = []
    for i in range(0, len(imgs2d), batch_size):
        x = torch.cat([L.to_tensor(im) for im in imgs2d[i:i + batch_size]], dim=0)
        out = model(pixel_values=x).last_hidden_state[:, 1:, :]  # без CLS
        outs.append(out)
    return torch.cat(outs, dim=0)


def anomaly_map(query_tokens: torch.Tensor, bank: torch.Tensor, k: int = 5) -> np.ndarray:
    """Для каждого патча — среднее косинусное расстояние до k ближайших патчей в банке."""
    q = F.normalize(query_tokens, dim=1)
    d = 1 - q @ bank.T
    s = d.topk(min(k, d.shape[1]), largest=False).values.mean(dim=1)
    return s.reshape(GRID, GRID).numpy()


def main():
    t_start = time.time()
    print("Загружаю данные...")
    air_all, _ = L.load_air(N_REF + 4)
    xe_all, _ = L.load_xe(4)
    g_lo, g_hi = L.compute_global_window([air_all])

    def phys(v):
        return np.clip((v - g_lo) / (g_hi - g_lo), 0, 1).astype(np.float32)

    ref_imgs = [L.prepare_2d(phys(v)) for v in air_all[:N_REF]]
    test_air = [L.prepare_2d(phys(v)) for v in air_all[N_REF:]]
    test_xe = [L.prepare_2d(phys(v)) for v in xe_all]

    backbone = L.load_dinov2("base", pooling="mean")
    model = backbone.model

    print(f"Строю банк патч-токенов из {N_REF} референсных срезов...")
    ref_tok = patch_tokens(model, ref_imgs)
    bank = F.normalize(ref_tok.reshape(-1, ref_tok.shape[-1]), dim=1)
    print(f"  банк: {bank.shape[0]} патч-векторов размерности {bank.shape[1]}")

    # Что показываем: чистый Air (норма), Xe (естественный сдвиг),
    # и Air с локальным искажением — проверка, находит ли карта именно ту область.
    cases = {}
    cases["air_clean"] = test_air[0]
    cases["xe_natural"] = test_xe[0]

    local = test_air[1].copy()
    h = SIZE // 2
    local[h - 56:h + 56, h - 56:h + 56] = L.corrupt_blur(local[h - 56:h + 56, h - 56:h + 56], 5)
    cases["air_local_blur_center"] = local

    local2 = test_air[2].copy()
    local2[:, SIZE // 2:] = np.clip(local2[:, SIZE // 2:] + 0.20, 0, 1)
    cases["air_right_half_brighter"] = local2

    cases["air_global_noise_s4"] = L.corrupt_noise(test_air[3], 4)

    print("Считаю карты аномальности...")
    maps, stats = {}, {}
    for name, img in cases.items():
        tok = patch_tokens(model, [img])[0]
        m = anomaly_map(tok, bank)
        maps[name] = m
        stats[name] = {"mean": float(m.mean()), "max": float(m.max()), "p95": float(np.percentile(m, 95))}
        print(f"  {name:26s} mean={m.mean():.4f} p95={np.percentile(m,95):.4f} max={m.max():.4f}")

    vmax = max(m.max() for m in maps.values())
    fig, axes = plt.subplots(2, len(cases), figsize=(3.1 * len(cases), 6.4))
    for j, (name, img) in enumerate(cases.items()):
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(name, fontsize=9)
        axes[0, j].axis("off")
        up = F.interpolate(torch.from_numpy(maps[name])[None, None].float(),
                           size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
        axes[1, j].imshow(img, cmap="gray")
        im = axes[1, j].imshow(up, cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
        axes[1, j].set_title(f"аномальность (mean={maps[name].mean():.3f})", fontsize=8)
        axes[1, j].axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="расстояние до банка нормы")
    fig.suptitle("Карта 'где модель сомневается': патч-токены DINOv2 vs банк нормальных срезов")
    fig.savefig(OUT / "spatial_uncertainty.png", dpi=150, bbox_inches="tight")

    with open(OUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    _write_note(stats)
    print(f"\nГотово за {time.time() - t_start:.0f}s -> {OUT}")


def _write_note(stats):
    base = stats["air_clean"]["mean"]
    lines = ["# Эксперимент 3: пространственная карта неопределённости", "",
             "Для каждого патча 14x14 считается расстояние до ближайших патчей",
             "в банке нормальных (Air) срезов. Чем ярче — тем менее знакомый участок.", "",
             f"Опорный уровень (чистый Air): mean={base:.4f}", "",
             "| Случай | mean | p95 | max | отношение mean к чистому Air |", "|---|---|---|---|---|"]
    for k, v in stats.items():
        lines.append(f"| {k} | {v['mean']:.4f} | {v['p95']:.4f} | {v['max']:.4f} | {v['mean']/base:.2f}x |")
    lines += ["", "Ключевая проверка — локальные случаи (`air_local_blur_center`,",
              "`air_right_half_brighter`): если карта подсвечивает именно искажённую область,",
              "а не всё изображение целиком, значит метод пространственно локализует проблему",
              "и его можно накладывать на φ-map как маску доверия."]
    (OUT / "note.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
