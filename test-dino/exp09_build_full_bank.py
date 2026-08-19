"""
Эксперимент 9: полный банк патч-векторов по всем срезам Сколтеха.

Идея разделения дорогого и дешёвого: прогон через сеть необратим и стоит десятки
минут, поэтому делается ОДИН раз для всех 900 срезов и кладётся на диск. Выбор
подмножества для работы детектора обратим и стоит секунды, поэтому крутится потом
сколько угодно раз без повторного прогона.

Нарезка банка плотнее, чем при проверке: сетка 3x3 тайла (покрывает 672x672
пикселя кадра) против 2x2 у `DomainGuard.check` (448x448). Банк и запрос не обязаны
иметь одинаковую нарезку — банк это просто набор эталонных патчей, с которыми
сравнивается запрос. Более плотное покрытие керна даёт больше материала для
последующих экспериментов при той же стоимости проверки.

Ограничение по памяти: 900 x 2304 x 768 в float16 это 3.2 ГБ на датасет, а
свободной оперативной памяти на машине ~6 ГБ. Поэтому банк пишется в memmap на
диск по одному срезу, и в оперативной памяти никогда не лежит целиком.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import torch

import lib_dino as L

OUT = L.RUNS_DIR / "bank_full"
GRID = 3          # 3x3 тайла по 224 пикселя = покрытие 672x672
TILE = 224
PATCHES = (TILE // 14) ** 2          # 256 патчей на тайл у ViT с патчем 14
TOKENS_PER_SLICE = GRID * GRID * PATCHES
PROGRESS = OUT / "progress.log"


def log(msg: str):
    print(msg)
    sys.stdout.flush()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@torch.no_grad()
def slice_tokens(model, img01: np.ndarray) -> np.ndarray:
    """Патч-токены одного среза, сетка GRIDxGRID тайлов в нативном разрешении."""
    h, w = img01.shape
    span = TILE * GRID
    r0, c0 = (h - span) // 2, (w - span) // 2
    tiles = [img01[r0 + i * TILE: r0 + (i + 1) * TILE, c0 + j * TILE: c0 + (j + 1) * TILE]
             for i in range(GRID) for j in range(GRID)]
    x = torch.cat([L.to_tensor(t) for t in tiles], dim=0)
    tok = model(pixel_values=x).last_hidden_state[:, 1:, :]     # (9, 256, D)
    return tok.reshape(-1, tok.shape[-1]).numpy().astype(np.float16)


def build(name: str, files: list, window: tuple[float, float], model, dim: int):
    n = len(files)
    tok_path = OUT / f"{name}_tokens.f16"
    hist_path = OUT / f"{name}_hist.npy"
    bank = np.memmap(tok_path, dtype=np.float16, mode="w+", shape=(n, TOKENS_PER_SLICE, dim))
    hists = np.zeros((n, 64), dtype=np.float32)

    lo, hi = window
    t0 = time.time()
    for i, f in enumerate(files):
        from PIL import Image
        raw = np.array(Image.open(f)).astype(np.float32)
        img01 = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
        bank[i] = slice_tokens(model, img01)
        hists[i] = L.features_histogram(L.prepare_2d(img01))
        if (i + 1) % 50 == 0 or i == n - 1:
            el = time.time() - t0
            log(f"  {name}: {i+1}/{n} срезов, {el:.0f}s, осталось ~{el/(i+1)*(n-i-1):.0f}s")
    bank.flush()
    del bank
    np.save(hist_path, hists)
    return tok_path.stat().st_size / 1e9


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")

    air_files = L.list_slice_files(L.AIR_DIR, "Air")
    xe_files = L.list_slice_files(L.XE_DIR, "Xe")
    log(f"Найдено срезов: Air={len(air_files)}, Xe={len(xe_files)}")

    # окно яркости считается по Air и применяется к обоим — так же, как во всех
    # предыдущих экспериментах, иначе банки будут в разных шкалах
    log("Считаю глобальное окно яркости по Air...")
    sample_idx = np.linspace(0, len(air_files) - 1, 40, dtype=int)
    from PIL import Image
    sample = np.stack([np.array(Image.open(air_files[i])).astype(np.float32) for i in sample_idx])
    window = L.compute_global_window([sample])
    log(f"  окно: {window[0]:.0f}..{window[1]:.0f}")

    backbone = L.load_dinov2("base", pooling="mean")
    model = backbone.model
    dim = model.config.hidden_size
    log(f"Модель загружена, размерность {dim}, токенов на срез {TOKENS_PER_SLICE}")
    log(f"Ожидаемый размер банка: {len(air_files) * TOKENS_PER_SLICE * dim * 2 / 1e9:.2f} ГБ на датасет\n")

    sizes = {}
    for name, files in (("air", air_files), ("xe", xe_files)):
        log(f"Строю банк '{name}'...")
        sizes[name] = build(name, files, window, model, dim)
        log(f"  готово, {sizes[name]:.2f} ГБ на диске\n")

    meta = {"grid": GRID, "tile": TILE, "patches_per_tile": PATCHES,
            "tokens_per_slice": TOKENS_PER_SLICE, "dim": dim,
            "n_air": len(air_files), "n_xe": len(xe_files),
            "window": list(window), "dtype": "float16",
            "model": "facebook/dinov2-base", "pooling_layer": "last",
            "note": "банк 3x3 тайла; DomainGuard.check по умолчанию режет запрос 2x2"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Всё готово за {time.time()-t_start:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
