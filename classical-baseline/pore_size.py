"""Объективная проверка морфологии: распределение пор по размерам.

Сравнивать 3D-картинки на глаз бессмысленно - виден один подкуб из многих,
в произвольной ориентации, с чужими настройками рендера. Проверяемая величина
одна: распределение разрешённых пор по размерам. В статье оно приведено на
Fig. 5 по данным ЯМР:

  - основная популяция пор имеет диаметр ~1-3 мкм, то есть НА УРОВНЕ вокселя
    (3 мкм) или мельче - это и есть субразрешённая пористость;
  - разрешённые поры - правый хвост за линией разрешения, с небольшим
    вторичным горбом в области 30-100 мкм.

Отсюда прямое следствие, которое снимает вопрос "почему у нас поры мельче,
чем на картинке в статье": в этой породе разрешённые поры ФИЗИЧЕСКИ малы,
типичное тело - считанные воксели в поперечнике. Толстые связные тела на их
Fig. 9(a) - это редкий крупный хвост распределения, а не типичная пора.

Размер меряется локальной толщиной (local thickness): для каждого воксельного
шара, вписанного в поровое пространство, его диаметр приписывается всем
вокселям шара. Это стандартная для цифрового керна мера, совпадающая по
смыслу с "диаметром поры" в ЯМР-распределении.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.io import imread

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

AIR = "hgxdh8ps94-1/Air/Air_%04d.png"
SUB = 200
VOXEL_UM = 3.0
POSITIONS = [(350, 350, 350), (150, 600, 200), (620, 200, 600)]


def local_thickness(mask, max_r=14):
    """Локальная толщина: диаметр наибольшего вписанного шара через воксель.

    Раскрытие шаром радиуса r равно множеству точек, удалённых не более чем на r
    от {dist >= r}. Через два преобразования расстояния это считается на порядок
    быстрее, чем прямой морфологической дилатацией шаром 2r+1.
    """
    dist = ndi.distance_transform_edt(mask)
    th = np.zeros(mask.shape, dtype=np.float32)
    for r in range(max_r, 0, -1):
        core = dist >= r
        if not core.any():
            continue
        grown = ndi.distance_transform_edt(~core) <= r
        th[grown & mask & (th == 0)] = 2.0 * r
    th[mask & (th == 0)] = 1.0            # воксели тоньше вписанного шара r=1
    return th


def load_cube(z0, y0, x0, n=SUB):
    return np.stack([imread(AIR % z).astype(np.float32)[y0:y0 + n, x0:x0 + n]
                     for z in range(z0, z0 + n)])


# Локальная толщина квантована: вписанный шар имеет целый радиус r, поэтому
# диаметр принимает только значения 2r вокселей. Границы корзин ставим между
# этими значениями, иначе часть корзин оказывается пустой чисто из-за сетки.
edges = (np.arange(1, 16) * 2.0 - 1.0) * VOXEL_UM
cent = np.sqrt(edges[1:] * edges[:-1])
acc = {"сырой порог": np.zeros(len(cent)), "median 3x3x3": np.zeros(len(cent))}
tot = {k: 0 for k in acc}

for (z0, y0, x0) in POSITIONS:
    v = load_cube(z0, y0, x0)
    for name, vol in (("сырой порог", v), ("median 3x3x3", ndi.median_filter(v, size=3))):
        m = vol < np.percentile(vol, 6.0)
        th = local_thickness(m) * VOXEL_UM
        h, _ = np.histogram(th[m], bins=edges)
        acc[name] += h
        tot[name] += int(m.sum())
    print(f"подкуб {(z0,y0,x0)}: обработан")

print(f"\n{'диаметр, мкм':>16} {'сырой порог':>14} {'median 3x3x3':>14}")
print("-" * 48)
for i, c in enumerate(cent):
    print(f"{edges[i]:6.1f}-{edges[i+1]:<6.1f} "
          f"{100*acc['сырой порог'][i]/tot['сырой порог']:>13.1f}% "
          f"{100*acc['median 3x3x3'][i]/tot['median 3x3x3']:>13.1f}%")

d50 = {}
for k in acc:
    c = np.cumsum(acc[k]) / acc[k].sum()
    d50[k] = float(np.interp(0.5, c, cent))
print(f"\nмедианный диаметр поры: сырой {d50['сырой порог']:.1f} мкм, "
      f"после median3 {d50['median 3x3x3']:.1f} мкм")
print(f"размер вокселя {VOXEL_UM:.0f} мкм; в статье (ЯМР, Fig. 5) основная")
print(f"популяция пор ~1-3 мкм, разрешённый хвост тянется до ~100 мкм")

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
for k, col in (("сырой порог", "C3"), ("median 3x3x3", "C0")):
    ax[0].step(cent, 100 * acc[k] / tot[k], where="mid", lw=2, color=col, label=k)
ax[0].axvline(VOXEL_UM, c="k", ls="--", lw=1.4, label=f"воксель {VOXEL_UM:.0f} мкм")
ax[0].axvspan(1, 2 * VOXEL_UM, color="grey", alpha=.18,
              label="ниже предела разрешения")
ax[0].axvspan(30, 100, color="green", alpha=.12, label="крупный хвост ЯМР (30-100 мкм)")
ax[0].set_xscale("log"); ax[0].set_xlabel("диаметр поры, мкм")
ax[0].set_ylabel("доля порового объёма, %")
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3, which="both")
ax[0].set_title("распределение разрешённых пор по размерам\n"
                "(3 подкуба 200³, локальная толщина)")

for k, col in (("сырой порог", "C3"), ("median 3x3x3", "C0")):
    c = np.cumsum(acc[k]) / acc[k].sum() * 100
    ax[1].step(cent, c, where="mid", lw=2, color=col, label=f"{k} (D50={d50[k]:.0f} мкм)")
ax[1].axvline(VOXEL_UM, c="k", ls="--", lw=1.4)
ax[1].axhline(50, c="grey", lw=1)
ax[1].set_xscale("log"); ax[1].set_xlabel("диаметр поры, мкм")
ax[1].set_ylabel("накопленная доля, %")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, which="both")
ax[1].set_title("накопленное распределение\nсырой порог смещает медиану вниз:\n"
                "мелкие тела — разбитые шумом куски, а не отдельные поры")
plt.tight_layout(); plt.savefig("results/fig6_pore_size.png", dpi=95)
json.dump({k: dict(hist=acc[k].tolist(), d50=d50[k]) for k in acc}
          | {"bin_edges_um": edges.tolist()},
          open("results/pore_size.json", "w"), indent=2)
print("saved results/fig6_pore_size.png, results/pore_size.json")
