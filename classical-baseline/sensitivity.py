"""Бюджет неопределённостей интегральной пористости.

Интегральная пористость линейна по входным средним:

    phi = ( alpha * <I_xe> + beta - <I_air> ) / dI100 * 100 %

поэтому чувствительность к каждому параметру считается точно, без повторных
проходов по объёму: достаточно один раз получить <I_air> и <I_xe> по всем
900 срезам. Денойзинг на средние не влияет (медианный фильтр их почти не
смещает), поэтому здесь он не нужен.

При изменении alpha параметры beta и dI100 пересчитываются согласованно:
обе величины привязаны к тем же опорным вокселям, и менять alpha в отрыве
от них означало бы сравнивать разные калибровки, а не оценивать одну.
"""
import json
import sys

import numpy as np
from skimage.io import imread

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

AIR = "hgxdh8ps94-1/Air/Air_%04d.png"
XE = "hgxdh8ps94-1/Xenon_/Xe_%04d.png"

cal = json.load(open("results/calibration.json"))
I_Q = cal["I_quartz"]
BETA0 = cal["beta"]
D100_0 = cal["d100"]
I_PORE = cal["I_pore"]
SIG_BETA = cal["beta_scatter"]

# сырые интенсивности опорных вокселей (до нормализации)
IXE_Q = I_Q - BETA0                       # кварц в Xe-скане
IXE_P = I_PORE + (D100_0 - BETA0)         # 100% пора в Xe-скане

print("считаю <I_air> и <I_xe> по всем 900 срезам...")
sa = sx = 0.0
for z in range(900):
    sa += float(imread(AIR % z).astype(np.float64).mean())
    sx += float(imread(XE % z).astype(np.float64).mean())
MA, MX = sa / 900, sx / 900
print(f"<I_air> = {MA:.1f}   <I_xe> = {MX:.1f}   сырая разность = {MX-MA:+.1f}\n")


def phi_of(alpha=1.0, beta=None, d100=None):
    """phi при заданной alpha; beta и dI100 пересчитываются по тем же опорам."""
    b = (I_Q - alpha * IXE_Q) if beta is None else beta
    d = (alpha * IXE_P + b - I_PORE) if d100 is None else d100
    return (alpha * MX + b - MA) / d * 100.0


base = phi_of()
print(f"базовая оценка: phi = {base:.2f} %\n")
print(f"{'параметр':<42} {'значение':>12} {'phi, %':>8} {'сдвиг':>8}")
print("-" * 74)


def row(name, val, phi):
    print(f"{name:<42} {val:>12} {phi:>8.2f} {phi-base:>+8.2f}")


row("база (alpha=1)", "1.0000", base)
row("alpha по опоре плотных включений", "0.9765", phi_of(alpha=0.9765))
row("alpha на столько же в другую сторону", "1.0235", phi_of(alpha=1.0235))
print()
row("beta +1 sigma разброса по срезам", f"{BETA0+SIG_BETA:+.1f}", phi_of(beta=BETA0 + SIG_BETA))
row("beta -1 sigma", f"{BETA0-SIG_BETA:+.1f}", phi_of(beta=BETA0 - SIG_BETA))
row("beta +30 (= 1 п.п. пористости)", f"{BETA0+30:+.1f}", phi_of(beta=BETA0 + 30))
print()
row("dI100 +5 %", f"{D100_0*1.05:.0f}", phi_of(d100=D100_0 * 1.05))
row("dI100 -5 %", f"{D100_0*0.95:.0f}", phi_of(d100=D100_0 * 0.95))
row("dI100 = медиана по внутренности пор", "3038", phi_of(d100=3038))
row("dI100 = экстраполяция гребня", "3176", phi_of(d100=3176))

# --- сводка ----------------------------------------------------------------
e_alpha = abs(phi_of(alpha=0.9765) - base)
e_beta = abs(phi_of(beta=BETA0 + SIG_BETA) - base)
e_d100 = abs(phi_of(d100=D100_0 * 1.05) - base)
tot = float(np.hypot(np.hypot(e_alpha, e_beta), e_d100))

print("\n" + "=" * 74)
print("СВОДКА")
print("=" * 74)
print(f"  вклад alpha (расхождение двух твёрдых опор) : +-{e_alpha:.2f} п.п.")
print(f"  вклад beta  (разброс опоры по срезам)       : +-{e_beta:.2f} п.п.")
print(f"  вклад dI100 (+-5 %)                          : +-{e_d100:.2f} п.п.")
print(f"  суммарно (квадратично)                      : +-{tot:.2f} п.п.")
print(f"\n  ИТОГ:  phi = {base:.2f} +- {tot:.2f} %")
print(f"  лаборатория 13.6 %  ->  расхождение "
      f"{abs(base-13.6):.2f} п.п. ({abs(base-13.6)/tot:.1f} sigma)")

json.dump(dict(phi=base, err_alpha=e_alpha, err_beta=e_beta,
               err_d100=e_d100, err_total=tot, mean_air=MA, mean_xe=MX),
          open("results/sensitivity.json", "w"), indent=2)
print("\nsaved results/sensitivity.json")
