"""Тесты пайплайна: вырожденные входы, инварианты, детерминизм, защита.

Проверяются не "похоже ли на правду", а свойства, которые обязаны выполняться
ТОЧНО, независимо от данных. Каждое из них ловит свой класс ошибок:

  вырожденные входы  - пайплайн должен возвращать известный ответ или падать
                       ГРОМКО, а не выдавать правдоподобный мусор;
  инварианты         - phi не должна зависеть от того, что физически на неё
                       не влияет (сдвиг и масштаб интенсивностей, шаг по z,
                       размер кропа, денойзинг);
  детерминизм        - два одинаковых прогона дают одинаковые числа;
  защита             - неподходящий вход отвергается, а не обрабатывается молча.

Запуск:  python tests.py            (полный набор, ~5 мин)
         python tests.py --fast     (без прогонов пайплайна, ~5 с)
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.io import imsave

sys.path.insert(0, str(Path(__file__).parent))
from xe_pipeline import compare_phi_maps

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

N = 128                      # ребро тестового фантома
I_QUARTZ, I_PORE, I_CLAY = 8860.0, 7070.0, 9980.0
D100, BETA, SIGMA = 3142.0, 265.0, 78.0

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'OK ' if ok else 'ПРОВАЛ'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def blobs(n, scale, seed):
    g = np.random.default_rng(seed).normal(size=(n,) * 3).astype(np.float32)
    g = ndi.gaussian_filter(g, scale)
    return (g - g.mean()) / g.std()


def phantom(n=N, seed=0, with_pores=True, micro_level=0.30):
    """Фантом с известной phi. Опорная фаза строго непориста."""
    solid = np.full((n,) * 3, I_QUARTZ, dtype=np.float32)
    clay = blobs(n, 4.0, seed + 1) > 0.55
    solid[clay] = I_CLAY
    solid[blobs(n, 2.0, seed + 3) > 3.2] = 13500.0

    resolved = (blobs(n, 1.8, seed + 4) > 1.85) if with_pores else np.zeros((n,)*3, bool)
    micro = np.clip(micro_level * (0.6 + np.abs(blobs(n, 5.0, seed + 5))), 0, 0.75)
    phi = np.where(resolved, 1.0, np.where(clay, micro, 0.0)).astype(np.float32)
    return phi, solid


def scans(phi, solid, beta=BETA, d100=D100, sigma=SIGMA, seed=11, gain=1.0, offset=0.0):
    """Синтезирует пару сканов. gain/offset - искажение ОБОИХ сканов сразу,
    имитирует другую шкалу реконструкции; phi от него зависеть не должна."""
    rng = np.random.default_rng(seed)
    air = phi * I_PORE + (1 - phi) * solid
    xe = air + phi * d100 - beta
    air = air + rng.normal(0, sigma, air.shape)
    xe = xe + rng.normal(0, sigma, xe.shape)
    air = np.clip(air * gain + offset, 0, 65535).astype(np.uint16)
    xe = np.clip(xe * gain + offset, 0, 65535).astype(np.uint16)
    return air, xe


def write(dirpath: Path, air, xe):
    (dirpath / "A").mkdir(parents=True, exist_ok=True)
    (dirpath / "X").mkdir(parents=True, exist_ok=True)
    for z in range(air.shape[0]):
        imsave(dirpath / "A" / f"a_{z:04d}.png", air[z], check_contrast=False)
        imsave(dirpath / "X" / f"x_{z:04d}.png", xe[z], check_contrast=False)


def run(dirpath: Path, nz, extra=()):
    out = dirpath / "out"
    cmd = [sys.executable, "xe_pipeline.py", "--stage", "all",
           "--air", str(dirpath / "A" / "a_%04d.png"),
           "--xe", str(dirpath / "X" / "x_%04d.png"),
           "--nz", str(nz), "--out", str(out), "--n-cal", "8", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None, None, r
    return (json.loads((out / "calibration.json").read_text(encoding="utf-8")),
            json.loads((out / "stats.json").read_text(encoding="utf-8")), r)


# ===========================================================================
def test_compare_phi_maps():
    print("\n--- compare_phi_maps: метрика для ML ---")
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, (4, 40, 40)).astype(np.float32)

    m = compare_phi_maps(a, a)
    check("одинаковые карты -> нулевая ошибка",
          m["mae"] == 0 and m["bias"] == 0 and abs(m["pearson"] - 1) < 1e-9,
          f"mae={m['mae']}, pearson={m['pearson']:.6f}")

    m = compare_phi_maps(a + 7.0, a)
    check("сдвиг на +7 -> bias=+7 и phi_abs_error=7",
          abs(m["bias"] - 7) < 1e-6 and abs(m["phi_abs_error"] - 7) < 1e-6,
          f"bias={m['bias']:.4f}")

    m = compare_phi_maps(a * 0.5 + 25, a)
    check("сжатие диапазона ловится по пористости, не по корреляции",
          m["pearson"] > 0.999 and m["phi_abs_error"] > 0.1,
          f"pearson={m['pearson']:.4f}, промах по phi={m['phi_abs_error']:.2f} п.п.")

    try:
        compare_phi_maps(a, a[:, :20])
        check("несовпадение форм -> исключение", False, "исключения не было")
    except ValueError:
        check("несовпадение форм -> исключение", True)


def test_degenerate(tmp: Path):
    print("\n--- вырожденные входы ---")
    # φ = 0 везде: измерять dI100 не по чему, пайплайн обязан УПАСТЬ громко
    phi, solid = phantom(with_pores=False, micro_level=0.0)
    d = tmp / "zero"
    write(d, *scans(phi, solid))
    cal, st, r = run(d, N)
    ok = cal is None and "разрешённых пор" in (r.stderr + r.stdout)
    check("объём без пористости -> явная ошибка, а не молчаливый мусор",
          ok, "упал с понятным сообщением" if ok else
              (f"вернул phi={st['total_porosity']:.2f}%" if st else "упал не с тем сообщением"))

    # известная постоянная микропористость
    phi, solid = phantom(seed=20)
    truth = float(phi.mean() * 100)
    d = tmp / "known"
    write(d, *scans(phi, solid))
    cal, st, r = run(d, N)
    if st is None:
        return check("известная phi восстанавливается", False, "пайплайн упал")
    err = st["total_porosity"] - truth
    check("известная phi восстанавливается (допуск 0.6 п.п.)", abs(err) < 0.6,
          f"истина {truth:.2f} %, оценка {st['total_porosity']:.2f} %, ошибка {err:+.2f} п.п.")
    check("beta восстанавливается (допуск 15 единиц)", abs(cal["beta"] - BETA) < 15,
          f"истина {BETA:.0f}, оценка {cal['beta']:.1f}")
    return st


def test_invariance(tmp: Path, base_st):
    print("\n--- инварианты: phi не должна зависеть от этого ---")
    phi, solid = phantom(seed=20)
    truth = float(phi.mean() * 100)

    # сдвиг обоих сканов на константу
    d = tmp / "shift"
    write(d, *scans(phi, solid, offset=1500.0))
    _, st, _ = run(d, N)
    check("сдвиг обоих сканов на +1500 не меняет phi",
          st is not None and abs(st["total_porosity"] - base_st["total_porosity"]) < 0.25,
          f"{base_st['total_porosity']:.2f} % -> {st['total_porosity']:.2f} %" if st else "упал")

    # масштаб обоих сканов
    d = tmp / "gain"
    write(d, *scans(phi, solid, gain=1.35))
    _, st, _ = run(d, N)
    check("масштаб обоих сканов x1.35 не меняет phi",
          st is not None and abs(st["total_porosity"] - base_st["total_porosity"]) < 0.25,
          f"{base_st['total_porosity']:.2f} % -> {st['total_porosity']:.2f} %" if st else "упал")

    # шаг по z: интегральная пористость не должна зависеть
    d = tmp / "known"
    _, st, _ = run(d, N, extra=("--zstep", "4"))
    check("шаг по z=4 не меняет интегральную phi",
          st is not None and abs(st["total_porosity"] - base_st["total_porosity"]) < 0.3,
          f"{base_st['total_porosity']:.2f} % -> {st['total_porosity']:.2f} %" if st else "упал")

    # Денойзинг - НЕ строгий инвариант, а параметр калибровки, и это надо
    # понимать. На средний dI он почти не влияет, но dI100 оценивается как
    # p90 - 1.2816*sigma, а sigma зависит от фильтра. Поэтому режим денойзинга
    # сдвигает шкалу пористости. На реальных данных разброс по четырём режимам
    # (none/median3/median5/gauss) составляет 0.28 п.п., на фантоме - до 1.2 п.п.
    # Тест сторожит, чтобы зависимость не выросла до неприемлемой.
    _, st, _ = run(d, N, extra=("--denoise", "none"))
    delta = abs(st["total_porosity"] - base_st["total_porosity"]) if st else 99
    check("режим денойзинга сдвигает phi не более чем на 1.5 п.п.", delta < 1.5,
          f"{base_st['total_porosity']:.2f} % -> {st['total_porosity']:.2f} % "
          f"(сдвиг {delta:.2f} п.п.)" if st else "упал")


def test_determinism(tmp: Path, base_st):
    print("\n--- детерминизм ---")
    _, st, _ = run(tmp / "known", N)
    check("повторный прогон даёт те же числа",
          st is not None and st["total_porosity"] == base_st["total_porosity"],
          f"{base_st['total_porosity']:.6f} vs {st['total_porosity']:.6f}" if st else "упал")


def test_guard():
    print("\n--- защита от неподходящего входа ---")
    src = Path("results/phi_map.npy")
    if not src.exists():
        return check("анизотропный объём отвергается make_3d", False, "нет results/phi_map.npy")
    shape = np.load(src, mmap_mode="r").shape
    check("phi_map на диске - полный изотропный объём", shape[0] == shape[1] == shape[2],
          f"форма {shape}" + ("" if shape[0] == shape[1] else "  <- z прорежен, связность считать нельзя"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="без прогонов пайплайна")
    args = ap.parse_args()

    print("=" * 70)
    print(f"ТЕСТЫ ПАЙПЛАЙНА   (фантом {N}^3)")
    print("=" * 70)

    test_compare_phi_maps()
    test_guard()

    if not args.fast:
        tmp = Path(tempfile.mkdtemp(prefix="xetest_"))
        try:
            base_st = test_degenerate(tmp)
            if base_st:
                test_invariance(tmp, base_st)
                test_determinism(tmp, base_st)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("\n(прогоны пайплайна пропущены: --fast)")

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"ИТОГ: {n_ok} из {len(RESULTS)} пройдено")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  ПРОВАЛ: {name}  {detail}")
    print("=" * 70)
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
