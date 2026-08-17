"""
Xe-enhanced micro-CT: классический (не-ML) референсный пайплайн.

Воспроизводит методологию Ebadi et al. (2022), "Lift the Veil of Secrecy in
Sub-Resolved Pores by Xe-Enhanced Computed Tomography", на датасете Skoltech
(Mendeley DOI 10.17632/hgxdh8ps94.1).

Опубликованные числа для этого образца (tight sandstone, Ачимовская свита):
    лаборатория      phi = 13.6 %   k = 0.104 mD
    классика uxCT    phi = 6 %
    Xe-enhanced      phi = 11.1 %   k = 0.059 mD

--------------------------------------------------------------------------
ФИЗИЧЕСКАЯ МОДЕЛЬ
--------------------------------------------------------------------------
Воксель крупнее поры содержит смесь твёрдой фазы и порового пространства
(partial volume effect). Для доли пор phi в вокселе:

    I_air = phi * I_pore_air + (1 - phi) * I_solid
    I_xe  = phi * I_pore_xe  + (1 - phi) * I_solid

Вычитание убирает вклад твёрдой фазы ТОЖДЕСТВЕННО:

    dI = I_xe - I_air = phi * (I_pore_xe - I_pore_air) = phi * dI100
    =>  phi = dI / dI100

В этом главный смысл метода, и он сильнее, чем просто "больше контраст".
В сухом скане интенсивность вокселя зависит и от пористости, и от минералогии
(кварц, полевой шпат, глина поглощают по-разному), и разделить два вклада
по одному скану нельзя. В разности минералогия сокращается точно, остаётся
только пористость.

Насколько это важно, видно на числах именно этого образца. Воксели, которые
метод относит к субразрешённой пористости (phi > 25 %, но не разрешённая
пора), имеют в сухом скане медиану I_air ~ 8800 при пике кварца 8860 -
разница 55 единиц при шуме сухого скана ~250. То есть в сухом скане они
неотличимы от чистого кварца. В разности те же воксели дают dI ~ 900 при
шуме разности 110, то есть сигнал 8 sigma. Причина в том, что твёрдая фаза
этих вокселей плотнее кварца (по двум независимым оценкам I_solid ~ 9400-9900,
что отвечает глинам с K и Fe), и её избыточное поглощение почти точно
компенсирует недостачу от пор. Сухой скан видит сумму двух эффектов и читает
её как обычную матрицу; разность видит только пористость.

--------------------------------------------------------------------------
АРХИТЕКТУРА: 2.5D streaming
--------------------------------------------------------------------------
phi - воксельная величина, в формуле выше нет ни одного 3D-соседа. Поэтому
объём обрабатывается срез за срезом с записью в memmap на диск: полный объём
900^3 никогда не оказывается в RAM целиком (два входных стека во float32 -
это 5.8 ГБ при 7.4 ГБ физической памяти на машине). Это позволяет считать
по ВСЕМУ образцу, а не по центральному кубику 256^3.

Глобальные константы (beta, пороги, dI100) считаются ОДИН раз по выборке
срезов. Если оценивать их на каждом срезе отдельно, калибровка поедет от z
к z и в объёме появится полосатость по z - артефакт обработки, на вид
неотличимый от геологической слоистости.

Регистрация не выполняется: кросс-корреляция z-профилей двух сканов даёт
сдвиг 0 срезов, разностная карта показывает связные поровые кластеры без
краевых диполей - объёмы уже совмещены на источнике.

--------------------------------------------------------------------------
Использование:
    python xe_pipeline.py --stage all              # калибровка + полный объём
    python xe_pipeline.py --stage run --zstep 10   # быстрая прикидка
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

AIR_PAT = "hgxdh8ps94-1/Air/Air_%04d.png"
XE_PAT = "hgxdh8ps94-1/Xenon_/Xe_%04d.png"
N_Z = 900

SE3 = np.ones((3, 3, 3), bool)
PHI_LO, PHI_HI, PHI_NB = -60.0, 200.0, 520     # сетка гистограммы phi


# ===========================================================================
# ввод-вывод
# ===========================================================================

def read_slice(pattern: str, z: int, crop: int | None = None) -> np.ndarray:
    img = imread(pattern % z)
    if img.ndim == 3:                      # PNG может оказаться RGB, каналы идентичны
        img = img[..., 0]
    img = img.astype(np.float32)
    if crop is not None and crop < img.shape[0]:
        y0 = (img.shape[0] - crop) // 2
        x0 = (img.shape[1] - crop) // 2
        img = img[y0:y0 + crop, x0:x0 + crop]
    return img


def read_block(pattern: str, z: int, crop=None) -> np.ndarray:
    """Три соседних среза - нужны для 3D-эрозии опорных масок."""
    return np.stack([read_slice(pattern, zz, crop) for zz in (z - 1, z, z + 1)])


def denoise_slice(a: np.ndarray, mode: str) -> np.ndarray:
    """Денойзинг РАЗНОСТИ, а не двух входов по отдельности.

    Шум в Air и Xe независим, поэтому один проход по разности вместо двух
    проходов по входам даёт тот же результат вдвое дешевле.

    Для интегральной пористости денойзинг не нужен: усреднение линейно, а
    медианный фильтр почти не смещает среднее. Он нужен для повоксельной
    phi-карты и для 3D-модели порового пространства.
    """
    if mode == "none":
        return a
    if mode == "median3":
        return ndi.median_filter(a, size=3)
    if mode == "median5":
        return ndi.median_filter(a, size=5)
    if mode == "gauss":
        return ndi.gaussian_filter(a, 1.0)
    raise ValueError(f"неизвестный режим денойзинга: {mode}")


# ===========================================================================
# ЭТАП A: калибровка по морфологически чистым опорным фазам
# ===========================================================================

# ПРОВЕРЕНО И ОТКЛОНЕНО: мода вместо медианы для опоры "кварц".
#
# Распределение dI по кварцевой опоре скошено вправо (масса справа/слева от
# моды = 1.75), потому что в окно +-200 попадает микропористая глина: её
# твёрдая фаза плотнее кварца, избыточное поглощение компенсирует недостачу
# от пор, и по сухому скану она садится ровно на пик. 3D-эрозия её не убирает,
# глинистые зоны пространственно связны. Теоретически при одностороннем хвосте
# мода предпочтительнее медианы.
#
# Но независимая опора на плотных включениях (I_air > 12000), где пористости
# заведомо нет, даёт beta = 277.5. Медиана по кварцу (271.9) согласуется с ней
# до 5.6 единиц, мода (300.7) расходится на 23. То есть загрязнение недостаточно
# велико, чтобы сломать медиану, а мода уводит в сторону. Разница стоит 1.07 п.п.
# пористости, поэтому она включена в бюджет неопределённостей как
# "неоднозначность оценки опоры" (+-0.9 п.п.). См. diagnostics/check_anchor.py.


def calibrate(sample_zs, crop, denoise_mode, out_dir: Path) -> dict:
    """Считает beta (нормализация) и dI100 (масштаб пористости).

    ПОЧЕМУ ОПОРЫ БЕРУТСЯ ЭРОЗИЕЙ, А НЕ ПО ГИСТОГРАММЕ.
    Любая оценка из гистограммы вблизи пика кварца загрязнена шумом: кварц
    занимает ~45% объёма при sigma сухого скана ~250, поэтому в корзине левее
    пика доминируют кварцевые воксели, случайно прочитавшиеся темнее, с
    истинным dI = 0. Именно это дало неверные alpha=0.80 (regression dilution)
    и невязку пористой ветви в +316 единиц при подгонке по условным медианам.
    Воксель, у которого все 26 соседей тоже лежат в окне опорной фазы,
    шумовым выбросом быть не может - согласованный выброс в 27 соседях
    невероятен. Отсюда 3D-эрозия опорных масок.

    Опоры:
      quartz - внутренность кварцевых зёрен, phi = 0  -> задаёт beta
      dense  - внутренность плотных включений, phi = 0 -> проверяет alpha
      pore   - внутренность крупных пор, phi = 1       -> задаёт dI100
    """
    print(f"=== ЭТАП A: калибровка по {len(sample_zs)} срезам ===")

    # --- A1: гистограмма сухого скана, пик кварца ---------------------------
    lo, hi, nb = 6000.0, 32000.0, 650
    edges = np.linspace(lo, hi, nb + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    h_air = np.zeros(nb, dtype=np.int64)
    for z in sample_zs:
        a = read_slice(AIR_PAT, z, crop)
        idx = np.clip((a - lo) / (hi - lo) * nb, 0, nb - 1).astype(np.int64)
        h_air += np.bincount(idx.ravel(), minlength=nb)

    i = int(h_air.argmax())
    y0, y1, y2 = h_air[i - 1], h_air[i], h_air[i + 1]
    I_quartz = float(cen[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (cen[1] - cen[0]))
    cum = np.cumsum(h_air) / h_air.sum() * 100.0

    # Порог опорных пор задаётся ФИЗИКОЙ, а не долей объёма.
    #
    # Перцентиль здесь не работает в принципе. Он привязан к тому, СКОЛЬКО в
    # образце разрешённых пор, а нужно попасть туда, ГДЕ они лежат по яркости.
    # Если поровая популяция узкая, перцентиль рассекает её посередине: маска
    # выходит крапчатой, 3D-эрозия не оставляет вокселей. Подбор перцентиля
    # "пока хоть что-то не останется" эту беду не лечит, а маскирует: на
    # тестовом фантоме он уезжал до p9, втягивал в опору микропористую глину и
    # занижал dI100 на 23 %, то есть менял громкое падение на тихое смещение
    # пористости на +3.5 п.п. (tests.py это ловит).
    #
    # Правильная привязка: сначала находим яркость самой тёмной фазы (это и
    # есть I_pore), затем берём воксели с phi > 75 %, то есть лежащие в первой
    # четверти пути от поры к кварцу. Такой порог не зависит от того, 3 % пор
    # в образце или 15 %, и всегда захватывает поры ЦЕЛИКОМ, а не их случайные
    # тёмные половинки - поэтому эрозия переживает его.
    #
    # Сухой скан перед пороговой обработкой обязательно денойзится: по сырым
    # данным порог сидит в ~1 sigma от границы и дробит связные тела в крошку
    # (diagnostics/check_segmentation.py: медианный объект 6 вокселей против 34).
    probe_zs = [sample_zs[len(sample_zs) // 4], sample_zs[len(sample_zs) // 2],
                sample_zs[3 * len(sample_zs) // 4]]
    probes = [ndi.median_filter(read_block(AIR_PAT, z, crop), size=3) for z in probe_zs]
    I_dark = float(np.percentile(np.concatenate([p.ravel() for p in probes]), 0.1))
    span = I_quartz - I_dark
    if span < 400:
        raise RuntimeError(
            f"в образце не видно разрешённых пор: самая тёмная фаза ({I_dark:.0f}) "
            f"отстоит от пика матрицы ({I_quartz:.0f}) всего на {span:.0f} единиц. "
            f"Измерять dI100 не по чему - его придётся задать извне, из давления "
            f"и состава газа.")

    # Минимальные размеры опор масштабируются с площадью среза, а не заданы
    # абсолютным числом: пороги, подобранные на срезах 900x900, на объёме
    # 128x128 недостижимы просто из-за площади, и калибровка падала бы на
    # исправных данных.
    area = probes[0].shape[1] * probes[0].shape[2]
    n_pore_min = max(60, int(3.0e-4 * area))
    n_quartz_min = max(200, int(2.0e-2 * area))
    n_dense_min = max(40, int(2.0e-4 * area))

    thr_dark, frac_used = None, None
    for frac in (0.25, 0.35, 0.50):          # phi > 75 %, > 65 %, > 50 %
        t = I_dark + frac * span
        n = np.mean([int(ndi.binary_erosion(b < t, SE3)[1].sum()) for b in probes])
        if n >= n_pore_min:
            thr_dark, frac_used = t, frac
            break
    if thr_dark is None:
        raise RuntimeError(
            f"опорные поры есть, но все тоньше 3 вокселей: после 3D-эрозии их "
            f"остаётся меньше {n_pore_min} на срез даже при пороге phi > 50 %. "
            f"dI100 по таким данным измерить нельзя.")
    del probes
    print(f"[A1] пик кварца: {I_quartz:.0f}    I_dark: {I_dark:.0f}    "
          f"порог опорных пор: {thr_dark:.0f} (phi > {100*(1-frac_used):.0f} %, "
          f"{float(np.interp(thr_dark, cen, cum)):.1f} % объёма)")

    # --- A2: опорные измерения ----------------------------------------------
    q_dI, d_dI, p_dI, p_air, negs, npore = [], [], [], [], [], []
    for z in sample_zs:
        a = read_block(AIR_PAT, z, crop)
        x = read_block(XE_PAT, z, crop)
        dI = np.stack([denoise_slice(x[k] - a[k], denoise_mode) for k in range(3)])

        # Маски строятся по ДЕНОЙЗЕННОМУ сухому скану той же медианой 3x3x3,
        # что и порог: по сырым данным порог сидит в ~1 sigma от границы и
        # режет связные тела в крошку, а эрозия затем стирает и её.
        a_dn = ndi.median_filter(a, size=3)
        q = ndi.binary_erosion(np.abs(a - I_quartz) < 200, SE3)[1]
        dn = ndi.binary_erosion(a > 12000, SE3)[1]
        p = ndi.binary_erosion(a_dn < thr_dark, SE3)[1]

        if q.sum() > n_quartz_min:
            q_dI.append(float(np.median(dI[1][q])))
        if dn.sum() > n_dense_min:
            d_dI.append(float(np.median(dI[1][dn])))
        if p.sum() > n_pore_min:
            p_dI.append(np.percentile(dI[1][p], 90))
            p_air.append(float(np.median(a[1][p])))
            npore.append(int(p.sum()))
        negs.append(dI[1][dI[1] < np.median(dI[1])])

    beta = -float(np.median(q_dI))
    beta_sd = float(np.std(q_dI))
    print(f"[A2] опора кварц:   dI_raw = {-beta:+.1f}  "
          f"(разброс по срезам {beta_sd:.1f} = {beta_sd/31:.2f}% пористости)")
    print(f"     -> beta = {beta:+.1f}")

    alpha_implied = None
    if d_dI:
        dd = float(np.median(d_dI))
        gap = dd + beta
        alpha_implied = 1.0 + gap / (13000.0 - I_quartz)
        print(f"[A2] опора плотные: dI = {gap:+.0f} после нормализации")
        print(f"     две твёрдые опоры разнесены по интенсивности на "
              f"{13000-I_quartz:.0f} единиц, расхождение {gap:+.0f}")
        # Расхождение опор в 97 единиц НЕ переносится в пористость один к одному:
        # оно действует пропорционально удалению от опорного кварца, а подавляющая
        # часть объёма сидит вблизи него. Точный вклад считает sensitivity.py:
        # +-0.29 п.п., а не 97/31 = 3.1 п.п., как дала бы наивная оценка.
        print(f"     -> неявное alpha = {alpha_implied:.4f}; принимаем alpha = 1 "
              f"(вклад в пористость +-0.3 п.п., см. sensitivity.py)")

    # --- A3: шум разности ----------------------------------------------------
    # phi >= 0 физически, поэтому левая половина распределения dI - чистый шум.
    neg = np.concatenate(negs)
    med = float(np.median(neg))
    sigma = float(np.median(np.abs(neg - med)) / 0.6745)
    print(f"[A3] шум dI после '{denoise_mode}': sigma = {sigma:.0f}")

    # --- A4: dI100 -----------------------------------------------------------
    # Две помехи действуют в разные стороны и снимаются по отдельности:
    #   частичное заполнение опоры матрицей ЗАНИЖАЕТ dI100 (эрозия на 2 вокселя
    #     стирает маску пор целиком - поры тоньше 3 вокселей, идеально чистой
    #     опоры в этих данных не существует), поэтому берём p90, а не медиану;
    #   шум РАЗДУВАЕТ верхний квантиль, поэтому вычитаем 1.2816*sigma - ровно
    #     смещение 90-го квантиля гауссова шума.
    p90 = float(np.median(p_dI)) + beta
    d100 = p90 - 1.2816 * sigma
    I_pore = float(np.median(p_air))
    print(f"[A4] опора поры: p90(dI) = {p90:.0f} по {int(np.median(npore))} вокс/срез")
    print(f"     dI100 = p90 - 1.2816*sigma = {d100:.0f}")
    print(f"     I_pore = {I_pore:.0f}, контраст кварц-пора в сухом = {I_quartz-I_pore:.0f}")
    print(f"     выигрыш Xe по контрасту: x{d100/(I_quartz-I_pore):.2f}")
    print(f"     sigma_phi = {100*sigma/d100:.1f} %,  1% пористости = {d100/100:.0f} единиц")

    thr_pore = 0.5 * (I_quartz + I_pore)
    print(f"[A5] порог разрешённых пор (phi=50%): I_air < {thr_pore:.0f} -> "
          f"{float(np.interp(thr_pore, cen, cum)):.2f} % объёма")

    cal = dict(alpha=1.0, beta=beta, beta_scatter=beta_sd, d100=d100,
               sigma_dI=sigma, sigma_phi=100 * sigma / d100,
               I_quartz=I_quartz, I_pore=I_pore, thr_pore=thr_pore,
               thr_dark=thr_dark, alpha_implied=alpha_implied,
               denoise=denoise_mode, crop=crop,
               sample_zs=[int(v) for v in sample_zs])
    (out_dir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"константы -> {out_dir/'calibration.json'}")
    return cal


# ===========================================================================
# ЭТАП B: потоковый проход по объёму
# ===========================================================================

def run_volume(cal: dict, zs, out_dir: Path):
    beta, d100 = cal["beta"], cal["d100"]
    thr_pore, crop, denoise_mode = cal["thr_pore"], cal["crop"], cal["denoise"]

    probe = read_slice(AIR_PAT, zs[0], crop)
    ny, nx = probe.shape
    nz = len(zs)
    print(f"\n=== ЭТАП B: потоковый проход, объём {nz}x{ny}x{nx} ===")
    print(f"phi-map -> uint8 memmap, {nz*ny*nx/1e6:.0f} МБ на диске")

    phi_mm = np.lib.format.open_memmap(out_dir / "phi_map.npy", mode="w+",
                                       dtype=np.uint8, shape=(nz, ny, nx))
    por_mm = np.lib.format.open_memmap(out_dir / "binary_pores_classic.npy", mode="w+",
                                       dtype=np.uint8, shape=(nz, ny, nx))

    phi_hist = np.zeros(PHI_NB, dtype=np.int64)
    phi_z, res_z = [], []
    for i, z in enumerate(zs):
        air = read_slice(AIR_PAT, z, crop)
        xe = read_slice(XE_PAT, z, crop)
        dI = denoise_slice(xe - air, denoise_mode) + beta

        phi = dI / d100 * 100.0
        pores = air < thr_pore

        # Интегральная пористость - БЕЗ клиппинга. Шум симметричен относительно
        # нуля и в среднем сокращается; обрезав отрицательные значения, мы
        # оставили бы только положительный хвост шума и систематически завысили
        # микропористость (при sigma_phi ~ 3% это даёт около +1 п.п.).
        phi_z.append(float(phi.mean()))
        res_z.append(100.0 * float(pores.mean()))

        idx = np.clip((phi - PHI_LO) / (PHI_HI - PHI_LO) * PHI_NB,
                      0, PHI_NB - 1).astype(np.int32)
        phi_hist += np.bincount(idx.ravel(), minlength=PHI_NB)

        # на диск - клиппованная карта: uint8 не хранит отрицательные значения,
        # а для 3D-модели отрицательная пористость смысла не имеет
        phi_mm[i] = np.clip(phi, 0, 100).astype(np.uint8)
        por_mm[i] = pores

        if i % 100 == 0 or i == nz - 1:
            print(f"  z={z:4d} [{i+1:4d}/{nz}]  phi={phi_z[-1]:6.2f}%  "
                  f"разрешённая={res_z[-1]:5.2f}%")

    phi_mm.flush(); por_mm.flush()
    del phi_mm, por_mm

    phi_z = np.array(phi_z); res_z = np.array(res_z)
    total, resolved = float(phi_z.mean()), float(res_z.mean())

    # --- почему в статье 11.1%, а в лаборатории 13.6% -----------------------
    # Гипотеза: их phi-map обнуляет воксели, не попавшие в сегментированную
    # "porous part", то есть срезает диффузный хвост слабой микропористости.
    # Считаем, сколько пористости остаётся, если обнулить всё ниже порога T.
    cen = np.linspace(PHI_LO, PHI_HI, PHI_NB + 1)[:-1] + (PHI_HI - PHI_LO) / PHI_NB / 2
    tot = phi_hist.sum()
    trunc = []
    for T in (0, 2, 5, 10, 15, 20, 25, 30, 40):
        keep = cen >= T
        trunc.append((T, float((phi_hist[keep] * cen[keep]).sum() / tot)))

    print("\n" + "=" * 68)
    print("РЕЗУЛЬТАТ")
    print("=" * 68)
    print(f"  полная пористость (Xe-enhanced) : {total:6.2f} %"
          f"   [лаборатория 13.6 %, статья 11.1 %]")
    print(f"    разрешённая (классика uxCT)   : {resolved:6.2f} %   [статья 6 %]")
    print(f"    субразрешённая                : {total-resolved:6.2f} %")
    print(f"  разброс по срезам               : {phi_z.std():6.2f} % (1 sigma)")
    print(f"  диапазон по z                   : {phi_z.min():.2f} .. {phi_z.max():.2f} %")

    print(f"\n  Если обнулять воксели со слабым сигналом (как делает сегментация")
    print(f"  'porous part'), интегральная пористость падает так:")
    for T, v in trunc:
        mark = "  <-- около 11.1% из статьи" if abs(v - 11.1) < 0.45 else ""
        print(f"    отбросить phi < {T:2d}% : {v:6.2f} %{mark}")

    stats = dict(total_porosity=total, resolved_porosity=resolved,
                 subresolution_porosity=total - resolved,
                 phi_std_z=float(phi_z.std()),
                 phi_per_slice=phi_z.tolist(), resolved_per_slice=res_z.tolist(),
                 truncation_curve=trunc, phi_hist=phi_hist.tolist(),
                 phi_hist_centers=cen.tolist(),
                 zs=[int(v) for v in zs], calibration=cal)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nсохранено: phi_map.npy, binary_pores_classic.npy, stats.json")
    return stats


# ===========================================================================
# метрика для сравнения с ML-моделями (Люди 2/3/5)
# ===========================================================================

def compare_phi_maps(predicted_phi: np.ndarray, real_phi: np.ndarray) -> dict:
    """Сравнивает предсказанную phi-карту с референсной.

    Это метрика "ошибка на итоговой физической величине", а не на изображении.
    Модель может дать отличный SSIM на предсказанном Xe-скане и промахнуться
    по пористости: SSIM оценивает структуру и контраст ЛОКАЛЬНО и нормирован,
    а phi зависит от АБСОЛЮТНОЙ величины скачка dI. Модель, воспроизводящая
    текстуру, но сжимающая динамический диапазон, получит хороший SSIM и
    неверную пористость. Здесь калибровка такая: 1% пористости = 31 единица
    интенсивности из 8860, то есть 0.35% шкалы - в этой задаче SSIM почти
    не чувствует то, ради чего всё делается.

    Ключевое число - phi_abs_error (промах по интегральной пористости).
    """
    p = np.asarray(predicted_phi, dtype=np.float64).ravel()
    r = np.asarray(real_phi, dtype=np.float64).ravel()
    if p.shape != r.shape:
        raise ValueError(f"формы не совпадают: {predicted_phi.shape} vs {real_phi.shape}")

    d = p - r
    return dict(
        mae=float(np.abs(d).mean()),
        rmse=float(np.sqrt((d ** 2).mean())),
        bias=float(d.mean()),
        phi_pred=float(p.mean()),
        phi_real=float(r.mean()),
        phi_abs_error=float(abs(p.mean() - r.mean())),
        phi_rel_error=float(abs(p.mean() - r.mean()) / max(r.mean(), 1e-9) * 100.0),
        pearson=float(np.corrcoef(p, r)[0, 1]),
    )


# ===========================================================================
def main():
    global AIR_PAT, XE_PAT, N_Z
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["calibrate", "run", "all"], default="all")
    ap.add_argument("--out", default="results")
    ap.add_argument("--crop", type=int, default=0,
                    help="центральное окно по XY (0 = весь кадр 900x900)")
    ap.add_argument("--zstep", type=int, default=1, help="каждый N-й срез")
    ap.add_argument("--denoise", default="median3",
                    choices=["none", "median3", "median5", "gauss"])
    ap.add_argument("--n-cal", type=int, default=16, help="срезов на калибровку")
    # Пути и размер объёма - параметры, а не константы: на новых данных
    # меняется только эта строка запуска, код трогать не нужно.
    ap.add_argument("--air", default=AIR_PAT, help="шаблон пути к сухим срезам, с %%04d")
    ap.add_argument("--xe", default=XE_PAT, help="шаблон пути к Xe-срезам, с %%04d")
    ap.add_argument("--nz", type=int, default=N_Z, help="число срезов")
    args = ap.parse_args()
    AIR_PAT, XE_PAT, N_Z = args.air, args.xe, args.nz

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    crop = args.crop if args.crop > 0 else None

    if args.stage in ("calibrate", "all"):
        cal = calibrate(np.linspace(6, N_Z - 7, args.n_cal).astype(int).tolist(),
                        crop, args.denoise, out_dir)
    else:
        cal = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))

    if args.stage in ("run", "all"):
        run_volume(cal, list(range(0, N_Z, args.zstep)), out_dir)


if __name__ == "__main__":
    main()
