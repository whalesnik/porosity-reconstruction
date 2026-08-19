"""
domain_guard — практический детектор "этим данным доверять нельзя".

Это не эксперимент, а готовый модуль для остальной команды. Складывает вместе то,
что показали exp01-exp05, и даёт два ответа на новый срез:

  1. score  — насколько срез непохож на обучающие данные (одно число);
  2. map    — где именно на срезе непохожесть (карта в координатах изображения).

Ключевой вывод экспериментов: один признак не решает задачу. Гистограммная
статистика ловит смену настроек сканера, но слепа к смене структуры породы;
DINOv2 наоборот. Поэтому guard считает ДВА независимых канала и не смешивает их:

  channel "acquisition" — сменился ли прибор/настройки (гистограмма яркостей).
      Срабатывание -> данные нужно перенормировать, но порода, скорее всего, та же.
  channel "structure"   — сменилась ли сама порода (патч-токены DINOv2).
      Срабатывание -> реконструкции доверять нельзя, нужна доменная адаптация.

Смешивать их в одно число вредно: реакция на смену яркости и реакция на незнакомую
породу требуют разных действий от команды.

Выбор конкретных признаков — прямо из exp05 (таблица селективности):

  acquisition = classic-histogram. AUROC 0.936 на помехах съёмки при 0.536 на
      block_shuffle, то есть максимально чувствителен к прибору и почти слеп к
      структуре. Для этой роли его "недостаток" — ровно то, что нужно.
  structure   = DINOv2, ПОСЛЕДНИЙ слой, вход тайлами в нативном разрешении.
      Селективность +0.193 — лучшая из проверенных. Ранние слои (L3) детектируют
      сильнее по сырому AUROC (exp04), но реагируют и на смену яркости
      (помехи 0.964), поэтому для канала структуры не годятся.

Вход тайлами критичен: при сжатии 900->224 тот же признак ловит изменение размера
пор с AUROC 0.805, при нативных тайлах — 1.000 (exp05).

Использование:
    guard = DomainGuard.fit(reference_slices)      # 30-50 срезов из обучающего набора
    r = guard.check(new_slice)
    print(r.acquisition, r.structure, r.verdict)
    plt.imshow(r.structure_map)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

import lib_dino as L


@dataclass
class GuardResult:
    acquisition: float          # z-оценка сдвига условий съёмки
    structure: float            # z-оценка структурного сдвига
    structure_map: np.ndarray   # карта (16x16) структурной непохожести
    verdict: str
    trust: float = 1.0          # 0..1, «насколько данные знакомы» (см. _trust)
    p_typical: float = 1.0      # непараметрический p-value (см. _p_conformal)
    details: dict = field(default_factory=dict)


class DomainGuard:
    """Пороги калибруются по самим референсным данным (кросс-валидацией внутри
    reference), чтобы не подбирать их вручную под конкретный образец."""

    def __init__(self, ref_stats, ref_tokens, mu_a, sd_a, mu_s, sd_s, backbone, window,
                 cal_a=None, cal_s=None):
        self.ref_stats = ref_stats
        self.ref_tokens = ref_tokens
        self.mu_a, self.sd_a = mu_a, sd_a
        self.mu_s, self.sd_s = mu_s, sd_s
        self.backbone = backbone
        self.window = window
        # сырые leave-one-out скоры эталона — нужны для непараметрической вероятности
        self.cal_a = np.asarray(cal_a) if cal_a is not None else None
        self.cal_s = np.asarray(cal_s) if cal_s is not None else None

    # ------------------------------------------------- вероятность доверия ---
    @staticmethod
    def _trust(z: float) -> float:
        """0..1: доля эталонных срезов, которые выглядели бы как минимум настолько же
        необычно. Нормировано так, что типичный срез даёт 1.0.

        ВАЖНО про интерпретацию: это вероятность того, что данные ЗНАКОМЫ модели,
        а не вероятность того, что предсказание реконструкции верно. Второе можно
        будет утверждать только после проверки, что ошибка реконструкции реально
        растёт вместе с этим числом — для этого нужна обученная модель Людей 2/3.

        Ниже ~0.02 число не стоит читать буквально: при 40 эталонных срезах хвост
        распределения не измерен, а экстраполирован, и означает просто «далеко за
        пределами известного»."""
        from math import erf, sqrt
        upper_tail = 0.5 * (1.0 - erf(z / sqrt(2.0)))   # P(Z >= z) при N(0,1)
        return float(min(1.0, 2.0 * upper_tail))

    def _p_conformal(self, raw: float, cal: np.ndarray | None) -> float:
        """Непараметрический p-value: какая доля эталонных срезов оказалась НЕ ближе
        к норме, чем проверяемый. Не опирается на нормальность распределения, но
        разрешение ограничено размером эталона (при 40 срезах шаг ~1/41 = 0.024)."""
        if cal is None or len(cal) == 0:
            return float("nan")
        return float((1 + int((cal >= raw).sum())) / (len(cal) + 1))

    # ------------------------------------------------------------------ fit ---
    @classmethod
    def fit(cls, ref_slices: list[np.ndarray], window: tuple[float, float] | None = None,
            model_size: str = "base") -> "DomainGuard":
        """ref_slices — сырые 2D-срезы (uint16 или float) из обучающего набора."""
        if window is None:
            window = L.compute_global_window([np.stack(ref_slices)])
        lo, hi = window
        norm = [np.clip((s - lo) / (hi - lo), 0, 1).astype(np.float32) for s in ref_slices]

        # канал съёмки — по сжатому изображению: гистограмма от разрешения не зависит
        hist = torch.from_numpy(np.stack([L.features_histogram(L.prepare_2d(v)) for v in norm])).float()
        # канал структуры — по тайлам в нативном разрешении (exp05)
        backbone = L.load_dinov2(model_size, pooling="mean")
        tokens = torch.stack([cls._slice_tokens(backbone, v) for v in norm])  # (N, T, D)
        bank = F.normalize(tokens.reshape(-1, tokens.shape[-1]), dim=1)

        a, s = cls._calibrate(hist, tokens, bank)
        return cls(hist, bank, a.mean(), a.std() + 1e-8, s.mean(), s.std() + 1e-8,
                   backbone, window, cal_a=a, cal_s=s)

    @staticmethod
    def _calibrate(hist: torch.Tensor, tokens: torch.Tensor, bank: torch.Tensor,
                   k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out скоры эталона: каждый срез оценивается относительно ОСТАЛЬНЫХ.
        Без этого пороги окажутся оптимистично занижены — срез всегда идеально
        похож сам на себя.

        Реализация через маскирование, а не пересборку банка. Наивный вариант
        (собирать банк заново для каждого среза) при 900 срезах означал бы 900
        пересборок матрицы на два миллиона векторов — часы вместо секунд."""
        n, t, _ = tokens.shape
        a_scores, s_scores = [], []
        q_all = F.normalize(tokens.reshape(-1, tokens.shape[-1]), dim=1)
        for i in range(n):
            others = torch.cat([hist[:i], hist[i + 1:]])
            a_scores.append(L.knn_score(hist[i:i + 1], others, k=k, metric="cosine").item())

            d = 1 - q_all[i * t:(i + 1) * t] @ bank.T          # (t, n*t)
            d[:, i * t:(i + 1) * t] = float("inf")             # свои патчи не считаются
            s_scores.append(float(d.topk(min(k, d.shape[1]), largest=False)
                                  .values.mean(dim=1).mean()))
        return np.array(a_scores), np.array(s_scores)

    # ---------------------------------------------------------------- check ---
    def check(self, slice_2d: np.ndarray, z_warn: float = 3.0, z_alarm: float = 6.0) -> GuardResult:
        lo, hi = self.window
        v = np.clip((slice_2d - lo) / (hi - lo), 0, 1).astype(np.float32)

        st = torch.from_numpy(L.features_histogram(L.prepare_2d(v))[None]).float()
        a_raw = L.knn_score(st, self.ref_stats, k=5, metric="cosine").item()
        a_z = (a_raw - self.mu_a) / self.sd_a

        tok = self._slice_tokens(self.backbone, v)
        per_patch = self._token_score(tok, self.ref_tokens)
        s_z = (float(per_patch.mean()) - self.mu_s) / self.sd_s
        g = int(np.sqrt(len(per_patch)))

        if s_z > z_alarm:
            verdict = "СТОП: структура породы незнакомая, реконструкции доверять нельзя"
        elif a_z > z_alarm:
            verdict = "ВНИМАНИЕ: другие условия съёмки, нужна перенормировка перед подачей в модель"
        elif s_z > z_warn or a_z > z_warn:
            verdict = "погранично: проверить вручную"
        else:
            verdict = "норма"

        s_raw = float(per_patch.mean())
        # доверие ведёт структурный канал: смена прибора лечится перенормировкой,
        # а незнакомая структура — нет, поэтому именно она ограничивает доверие
        return GuardResult(acquisition=a_z, structure=s_z,
                           structure_map=per_patch.reshape(g, g).numpy(),
                           verdict=verdict,
                           trust=self._trust(s_z),
                           p_typical=self._p_conformal(s_raw, self.cal_s),
                           details={"acquisition_raw": a_raw, "structure_raw": s_raw,
                                    "trust_acquisition": self._trust(a_z),
                                    "p_typical_acquisition": self._p_conformal(a_raw, self.cal_a)})

    # ------------------------------------------------------------ внутреннее ---
    @staticmethod
    def _grid_tiles(img01: np.ndarray, tile: int = 224, grid: int = 2) -> list[np.ndarray]:
        """Регулярная сетка тайлов из центра кадра, в НАТИВНОМ разрешении.
        Регулярная, а не случайная (как в exp05), чтобы карта была связной
        и результат воспроизводился от запуска к запуску."""
        h, w = img01.shape
        span = tile * grid
        r0, c0 = (h - span) // 2, (w - span) // 2
        return [img01[r0 + i * tile: r0 + (i + 1) * tile, c0 + j * tile: c0 + (j + 1) * tile]
                for i in range(grid) for j in range(grid)]

    @classmethod
    @torch.no_grad()
    def _slice_tokens(cls, backbone, img01: np.ndarray, tile: int = 224, grid: int = 2) -> torch.Tensor:
        """Патч-токены среза, сшитые из тайлов в связную сетку (grid*16, grid*16, D)."""
        tiles = cls._grid_tiles(img01, tile, grid)
        x = torch.cat([L.to_tensor(t) for t in tiles], dim=0)
        tok = backbone.model(pixel_values=x).last_hidden_state[:, 1:, :]  # (grid^2, 256, D)
        g = int(np.sqrt(tok.shape[1]))
        d = tok.shape[-1]
        tok = tok.reshape(grid, grid, g, g, d).permute(0, 2, 1, 3, 4).reshape(grid * g, grid * g, d)
        return tok.reshape(-1, d)

    @staticmethod
    def _token_score(tokens: torch.Tensor, bank: torch.Tensor, k: int = 5) -> torch.Tensor:
        q = F.normalize(tokens, dim=1)
        d = 1 - q @ bank.T
        return d.topk(min(k, d.shape[1]), largest=False).values.mean(dim=1)


def _demo():
    """Проверка на данных Сколтеха: норма / другая яркость / другая структура."""
    air, _ = L.load_air(44)
    ref, test = list(air[:40]), air[40:]
    guard = DomainGuard.fit(ref)
    lo, hi = guard.window

    def raw(img01):
        return img01 * (hi - lo) + lo

    base01 = np.clip((test[0] - lo) / (hi - lo), 0, 1).astype(np.float32)
    cases = {
        "чистый Air (контроль)": test[0],
        "чистый Air (другой срез)": test[1],
        "яркость +0.05": raw(np.clip(base01 + 0.05, 0, 1)),
        "яркость +0.20": raw(np.clip(base01 + 0.20, 0, 1)),
        "поры расширены (другая порода)": raw(L.structural_pore_dilate(base01, 3)),
        "поры сужены (другая порода)": raw(L.structural_pore_erode(base01, 3)),
        "блоки перемешаны": raw(L.structural_block_shuffle(base01, 3)),
    }
    xe, _ = L.load_xe(1)
    cases["Xe-скан"] = xe[0]

    print(f"\n{'случай':34s} {'acq_z':>8s} {'str_z':>8s}  вердикт")
    print("-" * 100)
    results = {}
    for name, sl in cases.items():
        r = guard.check(sl)
        results[name] = r
        print(f"{name:34s} {r.acquisition:8.2f} {r.structure:8.2f}  {r.verdict}")

    _plot_demo(results)


def _plot_demo(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = L.RUNS_DIR / "domain_guard_demo"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results)
    x = [results[n].acquisition for n in names]
    y = [results[n].structure for n in names]
    ax.axhspan(-5, 6, color="seagreen", alpha=0.07)
    ax.axvspan(-5, 6, color="seagreen", alpha=0.07)
    ax.axhline(6, color="crimson", ls="--", lw=1.5)
    ax.axvline(6, color="crimson", ls="--", lw=1.5)
    ax.scatter(x, y, s=70, c="steelblue", zorder=3)
    for n, xi, yi in zip(names, x, y):
        ax.annotate(n, (xi, yi), fontsize=7.5, xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("канал съёмки (z): сменился прибор →")
    ax.set_ylabel("канал структуры (z): сменилась порода →")
    ax.set_title("Два канала не путаются: смена яркости уходит вправо, смена породы — вверх")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "two_channels.png", dpi=140, bbox_inches="tight")

    interesting = ["чистый Air (контроль)", "яркость +0.20",
                   "поры расширены (другая порода)", "блоки перемешаны"]
    fig2, axes = plt.subplots(1, len(interesting), figsize=(3.2 * len(interesting), 3.6))
    vmax = max(results[n].structure_map.max() for n in interesting)
    for ax, n in zip(axes, interesting):
        im = ax.imshow(results[n].structure_map, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{n}\nstr_z={results[n].structure:.1f}", fontsize=8)
        ax.axis("off")
    fig2.colorbar(im, ax=axes.tolist(), shrink=0.8, label="структурная непохожесть")
    fig2.suptitle("Карта структурного канала (32x32 патча, тайлы в нативном разрешении)")
    fig2.savefig(out / "structure_maps.png", dpi=140, bbox_inches="tight")
    print(f"\nКартинки -> {out}")


if __name__ == "__main__":
    _demo()
