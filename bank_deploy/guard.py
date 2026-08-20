"""
Рабочий детектор: два числа по новому срезу.

Самодостаточный модуль. Нужны только этот файл, папка с банком рядом и
DINOv2-base (скачается при первом запуске или возьмётся из ../test-dino/models).

    from guard import Guard
    g = Guard()                        # ~5 с на загрузку
    r = g.check(slice_2d)              # ~0.7 с на срез
    print(r.acquisition_trust, r.structure_trust, r.verdict)

Что означают выдаваемые числа — см. README.md рядом. Коротко:

  acquisition_trust  0..1  насколько условия съёмки похожи на обучающие
  structure_trust    0..1  насколько порода похожа на обучающую

Это НЕ вероятность того, что предсказание пористости верно. Проверено на двух
образцах: связи скора с промахом по интегральной пористости нет (p от 0.19 до
0.92). Есть связь с локальной, попиксельной точностью, но слабая по величине:
переход от лучшей трети срезов к худшей меняет MAE на 4-10 %.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import erf, sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


@dataclass
class GuardResult:
    acquisition_z: float
    acquisition_trust: float
    structure_z: float
    structure_trust: float
    verdict: str
    structure_map: np.ndarray = field(default=None, repr=False)
    raw: dict = field(default_factory=dict, repr=False)


def _trust(z: float) -> float:
    """0..1, нормировано так, что типичный обучающий срез даёт 1.0.
    Ниже ~0.02 читать буквально нельзя: хвост распределения экстраполирован."""
    return float(min(1.0, 2.0 * (0.5 * (1.0 - erf(z / sqrt(2.0))))))


class Guard:
    def __init__(self, bank_dir: Path | str = HERE, model_dir: Path | str | None = None):
        bank_dir = Path(bank_dir)
        self.meta = json.loads((bank_dir / "meta.json").read_text(encoding="utf-8"))
        self.cal = json.loads((bank_dir / "calibration.json").read_text(encoding="utf-8"))

        n, per, dim = (self.meta["n_slices"], self.meta["patches_per_slice"], self.meta["dim"])
        raw = np.fromfile(bank_dir / "structure_bank.f16", dtype=np.float16).reshape(n, per, dim)
        self.bank = F.normalize(torch.from_numpy(raw.reshape(-1, dim).astype(np.float32)), dim=1)
        self.hist_bank = torch.from_numpy(np.load(bank_dir / "acquisition_bank.npy")).float()

        self.window = tuple(self.meta["window"])
        self.tile, self.grid, self.k = self.meta["tile"], self.meta["query_grid"], self.meta["knn_k"]

        from transformers import AutoModel
        src = "facebook/dinov2-base"
        if model_dir is None:
            local = HERE.parent / "test-dino" / "models" / "dinov2-base"
            if local.exists():
                src = str(local)
        else:
            src = str(model_dir)
        self.model = AutoModel.from_pretrained(src)
        self.model.eval()

    # ------------------------------------------------------------------ ввод ---
    def _to01(self, slice_2d: np.ndarray, window=None) -> np.ndarray:
        lo, hi = window or self.window
        return np.clip((slice_2d.astype(np.float32) - lo) / (hi - lo), 0, 1).astype(np.float32)

    def _tiles(self, img01: np.ndarray) -> list[np.ndarray]:
        h, w = img01.shape
        span = self.tile * self.grid
        if min(h, w) < span:
            raise ValueError(f"срез {h}x{w} меньше требуемых {span}x{span} пикселей")
        r0, c0 = (h - span) // 2, (w - span) // 2
        return [img01[r0 + i * self.tile:r0 + (i + 1) * self.tile,
                      c0 + j * self.tile:c0 + (j + 1) * self.tile]
                for i in range(self.grid) for j in range(self.grid)]

    @staticmethod
    def _hist(img01: np.ndarray, bins: int = 64) -> np.ndarray:
        small = torch.from_numpy(img01)[None, None]
        small = F.interpolate(small, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
        h, _ = np.histogram(small, bins=bins, range=(0, 1), density=True)
        return h.astype(np.float32)

    # ---------------------------------------------------------------- проверка ---
    @torch.no_grad()
    def check(self, slice_2d: np.ndarray, own_window: tuple[float, float] | None = None,
              z_warn: float = 3.0, z_alarm: float = 6.0) -> GuardResult:
        """slice_2d — сырой срез в единицах КТ, не меньше 448x448.

        own_window — если задан, структурный канал считается в этой шкале.
        Так снимается разница яркостных шкал между приборами, и связь с локальной
        точностью выходит заметно сильнее (rho +0.40 против +0.29 на чужом керне).
        Канал условий съёмки при этом всегда считается в шкале обучающих данных —
        иначе он по построению перестал бы видеть смену прибора."""
        img_ref = self._to01(slice_2d)

        h = torch.from_numpy(self._hist(img_ref)[None])
        a_raw = self._knn(h, self.hist_bank)
        a_z = (a_raw - self.cal["acquisition"]["mean"]) / (self.cal["acquisition"]["std"] + 1e-12)

        img_struct = self._to01(slice_2d, own_window) if own_window else img_ref
        x = torch.cat([self._t(t) for t in self._tiles(img_struct)], dim=0)
        tok = self.model(pixel_values=x).last_hidden_state[:, 1:, :]
        q = F.normalize(tok.reshape(-1, tok.shape[-1]), dim=1)
        d = 1 - q @ self.bank.T
        per_patch = d.topk(self.k, largest=False).values.mean(dim=1)
        s_raw = float(per_patch.mean())
        s_z = (s_raw - self.cal["structure"]["mean"]) / (self.cal["structure"]["std"] + 1e-12)

        if s_z > z_alarm:
            verdict = "порода незнакомая: локальная точность ожидаемо ниже, о пористости судить нельзя"
        elif a_z > z_alarm:
            verdict = "другие условия съёмки: перенормировать вход перед подачей в модель"
        elif s_z > z_warn or a_z > z_warn:
            verdict = "погранично: проверить вручную"
        else:
            verdict = "норма"

        return GuardResult(
            acquisition_z=a_z, acquisition_trust=_trust(a_z),
            structure_z=s_z, structure_trust=_trust(s_z),
            verdict=verdict,
            structure_map=self._stitch(per_patch),
            raw={"acquisition_raw": a_raw, "structure_raw": s_raw})

    def _stitch(self, per_patch: torch.Tensor) -> np.ndarray:
        """Склеивает патчи обратно в связную карту в координатах изображения.
        Тайлы шли по сетке grid x grid, внутри каждого патчи по p x p."""
        gr = self.grid
        p = int(np.sqrt(len(per_patch) // (gr * gr)))
        m = per_patch.reshape(gr, gr, p, p).permute(0, 2, 1, 3).reshape(gr * p, gr * p)
        return m.numpy()

    def _knn(self, query: torch.Tensor, ref: torch.Tensor) -> float:
        qn, rn = F.normalize(query, dim=1), F.normalize(ref, dim=1)
        d = 1 - qn @ rn.T
        return float(d.topk(min(self.k, d.shape[1]), largest=False).values.mean())

    @staticmethod
    def _t(img01: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img01)[None, None].repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std


def _demo():
    """Проверка на трёх заведомо разных случаях."""
    import pandas as pd
    from PIL import Image

    root = HERE.parent
    g = Guard()
    cases = []

    files = sorted((root / "hgxdh8ps94-1" / "Air").glob("Air_*.png"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    used = set(g.meta["slice_ids"])
    free = [i for i in range(len(files)) if i not in used]
    cases.append(("свой домен (Сколтех, не из банка)", np.array(Image.open(files[free[0]])), None))

    u = pd.read_csv(root / "ood_testset_univ" / "labels.csv").iloc[0]
    un = np.array(Image.open(root / "ood_testset_univ" / "input_dry" / u["id"])).astype(np.float32)
    cases.append(("чужой керн, как есть", un, None))
    cases.append(("чужой керн, шкала выровнена", un,
                  (float(np.percentile(un, 0.5)), float(np.percentile(un, 99.5)))))

    print(f"\n{'случай':38s} {'съёмка':>18s} {'порода':>18s}  вердикт")
    print("-" * 118)
    for name, img, win in cases:
        r = g.check(img, own_window=win)
        print(f"{name:38s} {r.acquisition_trust:8.3f} (z{r.acquisition_z:+7.1f}) "
              f"{r.structure_trust:8.3f} (z{r.structure_z:+6.1f})  {r.verdict}")


if __name__ == "__main__":
    _demo()
