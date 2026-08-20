"""
Эксперимент 18: перекалибровка порогов по ОТЛОЖЕННЫМ срезам.

Проблема, найденная при проверке рабочего модуля. Калибровка в exp17 считалась
leave-one-out по срезам самого банка. Но в работе детектор видит срезы, которых
в банке нет вообще, и для них распределение скора оказывается сдвинутым: на
15 сколтеховских срезах вне банка медиана z вышла +0.57 вместо нуля, и 3 из 15
знакомых срезов ложно превысили порог.

Причина проста: leave-one-out убирает из банка только собственные патчи среза,
но соседние по глубине срезы остаются, а они почти идентичны. Для нового среза
таких близнецов в банке может не быть.

Правильная калибровка — по срезам, которых в банке нет. Тогда шкала описывает
именно тот случай, ради которого детектор существует.
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

DEPLOY = L.PROJECT_ROOT / "bank_deploy"
N_CAL = 120          # сколько отложенных срезов брать на калибровку
K = 5


def log(msg):
    print(msg)
    sys.stdout.flush()


@torch.no_grad()
def main():
    t0 = time.time()
    meta = json.loads((DEPLOY / "meta.json").read_text(encoding="utf-8"))
    n, per, dim = meta["n_slices"], meta["patches_per_slice"], meta["dim"]
    tile, grid = meta["tile"], meta["query_grid"]
    lo, hi = meta["window"]

    raw = np.fromfile(DEPLOY / "structure_bank.f16", dtype=np.float16).reshape(n * per, dim)
    bank = F.normalize(torch.from_numpy(raw.astype(np.float32)), dim=1)
    hist_bank = torch.from_numpy(np.load(DEPLOY / "acquisition_bank.npy")).float()

    files = L.list_slice_files(L.AIR_DIR, "Air")
    used = set(meta["slice_ids"])
    free = [i for i in range(len(files)) if i not in used]
    ids = [free[i] for i in np.linspace(0, len(free) - 1, N_CAL, dtype=int)]
    log(f"Банк: {n} срезов | отложенных для калибровки: {len(ids)} из {len(free)} свободных")

    backbone = L.load_dinov2("base", pooling="mean")
    span = tile * grid
    s_scores, a_scores = [], []
    for c, sl in enumerate(ids):
        img = np.clip((np.array(Image.open(files[sl])).astype(np.float32) - lo) / (hi - lo), 0, 1)
        img = img.astype(np.float32)
        h, w = img.shape
        r0, c0 = (h - span) // 2, (w - span) // 2
        tiles = [img[r0 + i * tile:r0 + (i + 1) * tile, c0 + j * tile:c0 + (j + 1) * tile]
                 for i in range(grid) for j in range(grid)]
        x = torch.cat([L.to_tensor(t) for t in tiles], dim=0)
        tok = backbone.model(pixel_values=x).last_hidden_state[:, 1:, :]
        q = F.normalize(tok.reshape(-1, tok.shape[-1]), dim=1)
        d = 1 - q @ bank.T
        s_scores.append(float(d.topk(K, largest=False).values.mean(dim=1).mean()))

        hq = torch.from_numpy(L.features_histogram(L.prepare_2d(img))[None]).float()
        a_scores.append(L.knn_score(hq, hist_bank, k=K, metric="cosine").item())
        if (c + 1) % 30 == 0:
            log(f"  {c+1}/{len(ids)} ({time.time()-t0:.0f}s)")

    s, a = np.array(s_scores), np.array(a_scores)
    old = json.loads((DEPLOY / "calibration.json").read_text(encoding="utf-8"))

    log(f"\nструктура: было {old['structure']['mean']:.4f} +- {old['structure']['std']:.4f} (LOO по банку)")
    log(f"           стало {s.mean():.4f} +- {s.std():.4f} (отложенные срезы)")
    log(f"съёмка:    было {old['acquisition']['mean']:.6f} +- {old['acquisition']['std']:.6f}")
    log(f"           стало {a.mean():.6f} +- {a.std():.6f}")

    # проверка: сколько отложенных срезов теперь ложно срабатывает
    z = (s - s.mean()) / s.std()
    from math import erf, sqrt
    trust = np.array([min(1.0, 2 * (0.5 * (1 - erf(v / sqrt(2))))) for v in z])
    log(f"\nЛожных срабатываний на отложенных при пороге 0.05: "
        f"{int((trust < 0.05).sum())}/{len(trust)} ({100*(trust<0.05).mean():.0f} %)")
    log(f"Медиана z по построению 0, разброс {z.std():.2f}")

    calib = {"structure": {"mean": float(s.mean()), "std": float(s.std()),
                           "holdout_scores": s.tolist()},
             "acquisition": {"mean": float(a.mean()), "std": float(a.std()),
                             "holdout_scores": a.tolist()},
             "method": "калибровка по отложенным срезам, не входящим в банк",
             "n_calibration_slices": len(ids),
             "calibration_slice_ids": [int(i) for i in ids],
             "previous_loo": {"structure": [old["structure"]["mean"], old["structure"]["std"]],
                              "acquisition": [old["acquisition"]["mean"], old["acquisition"]["std"]]}}
    (DEPLOY / "calibration.json").write_text(json.dumps(calib, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    log(f"\nГотово за {time.time()-t0:.0f}s -> {DEPLOY / 'calibration.json'}")


if __name__ == "__main__":
    main()
