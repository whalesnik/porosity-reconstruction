"""Сквозная проверка пайплайна на синтетике с ИЗВЕСТНЫМ ответом.

Зачем. До сих пор мы сверялись с лабораторными 13.6 %, у которых своя
погрешность, да и меряют они другой объём (весь керн, а не поле зрения).
Совпадение приятно, но оно не отвечает на вопрос "какова собственная ошибка
метода". Синтетика отвечает: здесь phi каждого вокселя задана нами, поэтому
известны и beta, и dI100, и интегральная пористость.

Фантом воспроизводит ровно те особенности, на которых пайплайн ломался:
  - унимодальная гистограмма с доминирующим пиком кварца (~45 % объёма);
  - шум того же уровня, что в реальных данных (sigma = 78 при контрасте
    кварц-пора 1789), из-за которого гистограммные оценки смещены;
  - ГЛАВНОЕ: микропористая фаза, твёрдая основа которой ПЛОТНЕЕ кварца.
    В сухом скане такие воксели неотличимы от матрицы - именно их метод и
    обязан найти. Если пайплайн их теряет, синтетика это покажет.
  - неизвестный аппаратный сдвиг между сканами, который надо восстановить.

Скрипт пишет фантом обычными 16-битными PNG и запускает НЕИЗМЕНЁННЫЙ
xe_pipeline.py на них. То есть он же служит шаблоном применения к новым
данным: меняются только --air/--xe/--nz.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.io import imsave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("synthetic")
N = 200                      # ребро фантома
RNG = np.random.default_rng(7)

# --- истина, которую пайплайн обязан восстановить ---------------------------
I_QUARTZ = 8860.0
I_PORE = 7070.0
I_FELDSPAR = 9250.0
I_CLAY = 9980.0              # твёрдая глина ПЛОТНЕЕ кварца - главная ловушка.
                             # Значение подобрано так, чтобы избыточное поглощение
                             # глины почти точно компенсировало недостачу от пор,
                             # как это намерено на реальных данных: там скрытые
                             # воксели отстоят от пика кварца всего на ~55 единиц.
I_DENSE = 13500.0
D100_TRUE = 3142.0           # скачок в вокселе, полностью занятом Xe
BETA_TRUE = 265.0            # аппаратный сдвиг Xe-скана (пайплайн его не знает)
SIGMA = 78.0                 # шум одного скана, как в реальных данных


def blobs(shape, scale, seed):
    """Гладкое случайное поле - основа для геологических форм."""
    g = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    g = ndi.gaussian_filter(g, scale)
    return (g - g.mean()) / g.std()


print(f"строю фантом {N}^3 ...")
# --- минеральный каркас ------------------------------------------------------
f1, f2 = blobs((N,) * 3, 4.0, 1), blobs((N,) * 3, 7.0, 2)
solid = np.full((N,) * 3, I_QUARTZ, dtype=np.float32)
solid[f2 > 0.7] = I_FELDSPAR
clay_zone = f1 > 0.55                       # где живёт микропористая глина
solid[clay_zone] = I_CLAY
dense = blobs((N,) * 3, 2.0, 3) > 3.2
solid[dense] = I_DENSE

# --- пористость --------------------------------------------------------------
# разрешённые поры: компактные тела, медианный диаметр ~4 вокселя,
# как показало распределение по размерам на реальных данных
pore_field = blobs((N,) * 3, 1.8, 4)
resolved = pore_field > 1.85
phi_true = resolved.astype(np.float32)

# Субразрешённая пористость живёт только в глинистой фазе. Кварцевый фон
# делаем СТРОГО непористым, и это принципиально, а не косметика.
#
# В первой версии фантома фон имел phi ~ 1.6 %, и пайплайн занизил beta ровно
# на 50 единиц = 1.6 % пористости. Это не ошибка пайплайна: метод меряет
# пористость ОТНОСИТЕЛЬНО опорной фазы, поэтому пористость, равномерно
# присутствующая в самой опоре, невидима и вычитается из ответа. Ограничение
# фундаментальное, оно записано в README; здесь опора должна быть чистой,
# иначе тест проверяет не то, что нужно.
micro = np.clip(0.18 + 0.30 * blobs((N,) * 3, 5.0, 5), 0.0, 0.75).astype(np.float32)
phi_true = np.where(resolved, 1.0, np.where(clay_zone, micro, 0.0))
phi_true = np.clip(phi_true, 0.0, 1.0).astype(np.float32)

PHI_TRUE = float(phi_true.mean() * 100)
print(f"  истинная пористость фантома: {PHI_TRUE:.2f} %")
print(f"    разрешённая (phi=1): {100*resolved.mean():.2f} %")
print(f"    субразрешённая:      {PHI_TRUE - 100*resolved.mean():.2f} %")

# --- формирование сканов -----------------------------------------------------
i_air = phi_true * I_PORE + (1 - phi_true) * solid
i_xe = i_air + phi_true * D100_TRUE
i_xe_measured = i_xe - BETA_TRUE                     # аппаратный дрейф

air = np.clip(i_air + RNG.normal(0, SIGMA, i_air.shape), 0, 65535).astype(np.uint16)
xe = np.clip(i_xe_measured + RNG.normal(0, SIGMA, i_xe.shape), 0, 65535).astype(np.uint16)

# проверка, что ловушка действительно расставлена
hidden = clay_zone & ~resolved & (phi_true > 0.25)
print(f"  скрытых вокселей (микропора в плотной глине): {100*hidden.mean():.1f} % объёма")
print(f"    их медиана I_air = {np.median(air[hidden]):.0f} при пике кварца {I_QUARTZ:.0f}")
print(f"    -> в сухом скане они на {abs(np.median(air[hidden])-I_QUARTZ):.0f} единиц от кварца "
      f"при шуме {SIGMA:.0f}: неотличимы")

(OUT / "Air").mkdir(parents=True, exist_ok=True)
(OUT / "Xe").mkdir(parents=True, exist_ok=True)
for z in range(N):
    imsave(OUT / "Air" / f"Air_{z:04d}.png", air[z], check_contrast=False)
    imsave(OUT / "Xe" / f"Xe_{z:04d}.png", xe[z], check_contrast=False)
np.save(OUT / "phi_true.npy", (phi_true * 100).astype(np.float32))
print(f"  фантом записан в {OUT}/")

# --- прогон НЕИЗМЕНЁННОГО пайплайна -----------------------------------------
print("\nзапускаю xe_pipeline.py на синтетике...\n" + "=" * 62)
cmd = [sys.executable, "xe_pipeline.py", "--stage", "all",
       "--air", str(OUT / "Air" / "Air_%04d.png"),
       "--xe", str(OUT / "Xe" / "Xe_%04d.png"),
       "--nz", str(N), "--out", str(OUT / "results"), "--n-cal", "12"]
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-2600:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:]); sys.exit(1)

# --- сверка с истиной --------------------------------------------------------
cal = json.load(open(OUT / "results" / "calibration.json"))
st = json.load(open(OUT / "results" / "stats.json"))
phi_est = np.load(OUT / "results" / "phi_map.npy", mmap_mode="r")
phi_ref = np.load(OUT / "phi_true.npy", mmap_mode="r")

sys.path.insert(0, ".")
from xe_pipeline import compare_phi_maps
m = compare_phi_maps(np.array(phi_est[::4]).astype(np.float32),
                     np.array(phi_ref[::4]))

print("\n" + "=" * 62)
print("СВЕРКА С ИСТИНОЙ")
print("=" * 62)
rows = [
    ("beta (сдвиг сканов)", BETA_TRUE, cal["beta"], "единиц"),
    ("dI100 (масштаб phi)", D100_TRUE, cal["d100"], "единиц"),
    ("I_quartz", I_QUARTZ, cal["I_quartz"], "единиц"),
    ("I_pore", I_PORE, cal["I_pore"], "единиц"),
    ("полная пористость", PHI_TRUE, st["total_porosity"], "%"),
]
print(f"{'величина':>22} {'истина':>10} {'оценка':>10} {'ошибка':>12}")
print("-" * 60)
for name, t, e, unit in rows:
    err = e - t
    rel = f"{err:+.2f} ({100*err/t:+.1f} %)" if unit != "%" else f"{err:+.2f} п.п."
    print(f"{name:>22} {t:>10.1f} {e:>10.1f} {rel:>12}")

print(f"\nповоксельно (каждый 4-й срез):")
print(f"  MAE   = {m['mae']:.2f} п.п.")
print(f"  RMSE  = {m['rmse']:.2f} п.п.")
print(f"  смещение = {m['bias']:+.2f} п.п.")
print(f"  Pearson  = {m['pearson']:.3f}")

json.dump(dict(truth=dict(beta=BETA_TRUE, d100=D100_TRUE, phi=PHI_TRUE),
               estimated=dict(beta=cal["beta"], d100=cal["d100"],
                              phi=st["total_porosity"]),
               voxelwise=m), open("results/synthetic_test.json", "w"), indent=2)
print("\nsaved results/synthetic_test.json")
