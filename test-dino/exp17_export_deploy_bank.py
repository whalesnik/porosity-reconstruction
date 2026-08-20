"""
Эксперимент 17: сборка банка для прикладного применения.

Полный банк из exp09 (6 ГБ, 900 сухих + 900 ксеноновых срезов) нужен был для
исследования: чтобы прореживать его как угодно и проверять, сколько срезов
достаточно. Для работы столько не нужно:

  - exp10 показал, что качество шкалы выходит на потолок к 200 срезам одного
    образца, дальше насыщение;
  - exp16 показал, что ксеноновый банк для различения доменов бесполезен
    (AUROC 0.460), поэтому в рабочую поставку он не входит.

Итог: 200 срезов x 200 патчей = 40 000 векторов, 61 МБ в float16.

Что кладётся в поставку:
  structure_bank.f16   патч-векторы DINOv2 (канал породы)
  acquisition_bank.npy гистограммы яркости (канал условий съёмки)
  calibration.json     среднее и разброс по обоим каналам + сырые LOO-скоры
  meta.json            все константы, нужные для воспроизведения

Калибровка считается leave-one-out и ТОЙ ЖЕ процедурой, которой будут считаться
рабочие запросы (сетка 2x2 тайла), иначе пороги окажутся смещены: банк собран
сеткой 3x3, и если калибровать по ней же, рабочий скор будет систематически
отличаться от калибровочного.
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
OUT = L.PROJECT_ROOT / "bank_deploy"
N_SLICES, PER_SLICE, K = 200, 200, 5
TILE, GRID = 224, 2


def log(msg: str):
    print(msg)
    sys.stdout.flush()


@torch.no_grad()
def query_tokens(model, img01: np.ndarray) -> torch.Tensor:
    """Патч-токены запроса: сетка 2x2 тайла из центра, нативное разрешение."""
    h, w = img01.shape
    span = TILE * GRID
    r0, c0 = (h - span) // 2, (w - span) // 2
    tiles = [img01[r0 + i * TILE:r0 + (i + 1) * TILE, c0 + j * TILE:c0 + (j + 1) * TILE]
             for i in range(GRID) for j in range(GRID)]
    x = torch.cat([L.to_tensor(t) for t in tiles], dim=0)
    tok = model(pixel_values=x).last_hidden_state[:, 1:, :]
    return F.normalize(tok.reshape(-1, tok.shape[-1]), dim=1)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((BANK_DIR / "meta.json").read_text(encoding="utf-8"))
    lo, hi = meta["window"]

    air = np.memmap(BANK_DIR / "air_tokens.f16", dtype=np.float16, mode="r",
                    shape=(meta["n_air"], meta["tokens_per_slice"], meta["dim"]))
    hist_all = np.load(BANK_DIR / "air_hist.npy")

    ids = np.sort(np.random.default_rng(0).permutation(meta["n_air"])[:N_SLICES])
    log(f"Отобрано {len(ids)} срезов из {meta['n_air']}")

    struct = np.stack([np.asarray(air[i][:PER_SLICE], dtype=np.float16) for i in ids])
    hist = hist_all[ids].astype(np.float32)
    struct.tofile(OUT / "structure_bank.f16")
    np.save(OUT / "acquisition_bank.npy", hist)
    log(f"Банк структуры: {struct.shape} -> {struct.nbytes/1e6:.1f} МБ")
    log(f"Банк съёмки:    {hist.shape} -> {hist.nbytes/1e6:.3f} МБ")

    # --- калибровка теми же средствами, что и рабочий запрос ---
    log("Калибрую leave-one-out (сеткой 2x2, как рабочие запросы)...")
    backbone = L.load_dinov2("base", pooling="mean")
    files = L.list_slice_files(L.AIR_DIR, "Air")
    from PIL import Image

    flat = F.normalize(torch.from_numpy(struct.reshape(-1, struct.shape[-1]).astype(np.float32)), dim=1)
    hist_t = torch.from_numpy(hist)

    cal_s, cal_a = [], []
    for j, sl in enumerate(ids):
        raw = np.array(Image.open(files[sl])).astype(np.float32)
        img01 = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)

        q = query_tokens(backbone.model, img01)
        dist = 1 - q @ flat.T
        dist[:, j * PER_SLICE:(j + 1) * PER_SLICE] = float("inf")   # свои патчи не считаются
        cal_s.append(float(dist.topk(K, largest=False).values.mean(dim=1).mean()))

        h = torch.from_numpy(L.features_histogram(L.prepare_2d(img01))[None]).float()
        others = torch.cat([hist_t[:j], hist_t[j + 1:]])
        cal_a.append(L.knn_score(h, others, k=K, metric="cosine").item())

        if (j + 1) % 50 == 0:
            log(f"  {j+1}/{len(ids)} ({time.time()-t0:.0f}s)")

    cal_s, cal_a = np.array(cal_s), np.array(cal_a)
    calib = {"structure": {"mean": float(cal_s.mean()), "std": float(cal_s.std()),
                           "loo_scores": cal_s.tolist()},
             "acquisition": {"mean": float(cal_a.mean()), "std": float(cal_a.std()),
                             "loo_scores": cal_a.tolist()}}
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"  структура:  {cal_s.mean():.4f} +- {cal_s.std():.4f}")
    log(f"  съёмка:     {cal_a.mean():.6f} +- {cal_a.std():.6f}")

    out_meta = {"source": "hgxdh8ps94-1 / Air (Сколтех), сухие срезы",
                "n_slices": int(N_SLICES), "patches_per_slice": int(PER_SLICE),
                "n_vectors": int(N_SLICES * PER_SLICE), "dim": int(meta["dim"]),
                "dtype": "float16", "slice_ids": ids.tolist(),
                "model": "facebook/dinov2-base", "layer": "last", "pooling": "patch tokens",
                "window": [lo, hi], "query_grid": GRID, "tile": TILE, "knn_k": K,
                "note": "банк собран сеткой 3x3, запрос считается сеткой 2x2; "
                        "калибровка сделана процедурой запроса"}
    (OUT / "meta.json").write_text(json.dumps(out_meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"\nГотово за {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
