"""3D-визуализация в стиле статьи Ebadi et al. (2022), Fig. 9(a) и Fig. 9(d).

Их картинки сделаны в Avizo объёмным рендерингом, а не изоповерхностями:

  Fig. 9(a) "triple media" - синие НЕПРОЗРАЧНЫЕ затенённые изоповерхности
      разрешённых пор поверх красного ПОЛУПРОЗРАЧНОГО облака субразрешённой
      фазы. Твёрдая фаза не рисуется вовсе.
  Fig. 9(d) "3D phi-map" - объёмный рендер phi в градациях серого. Фон внутри
      бокса светлый именно потому, что прозрачность растёт вместе с phi:
      слабо пористый материал почти прозрачен и даёт лишь серую дымку, а
      воксели с высокой phi выходят плотными белыми телами.

Чтобы сравнение было честным, воспроизводятся и их соглашения:
  - размер куба ~120 вокселей (360 мкм) - измерено по линейке 50 мкм на Fig. 9;
  - синяя фаза = Binary uxCT статьи, то есть порог сухого скана, дающий 6 %
    объёма (у нашего порога phi=50 % получается 3.4 %, и куб выглядел бы
    заметно беднее, чем у них, по причине соглашения, а не физики).

Линейка считается точно: при параллельной проекции camera.parallel_scale -
это половина высоты кадра в мировых единицах, отсюда пикселей на микрон.

Запуск:  python render_3d.py
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage.io import imread

import pyvista as pv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

pv.OFF_SCREEN = True

SUB = 120               # ребро куба, вокселей (как в статье, по линейке)
VOXEL_UM = 3.0          # разрешение из статьи
WIN = 1000              # размер кадра, px
AIR = "hgxdh8ps94-1/Air/Air_%04d.png"

pm = json.load(open("results/paper_method.json"))
THR_6 = pm["thr_resolved_6pct"]

phi_mm = np.load("results/phi_map.npy", mmap_mode="r")
nz, ny, nx = phi_mm.shape
z0, y0, x0 = (nz - SUB) // 2, (ny - SUB) // 2, (nx - SUB) // 2
phi = np.array(phi_mm[z0:z0 + SUB, y0:y0 + SUB, x0:x0 + SUB]).astype(np.float32)

# Разрешённые поры по соглашению статьи (6 % объёма), а не по нашему phi=50 %.
#
# ДЕНОЙЗИНГ ЗДЕСЬ ОБЯЗАТЕЛЕН. Порог сухого скана попадает всего в ~1.1 sigma
# от границы (шум 78, контраст кварц-пора 1789), поэтому по сырым данным
# связные поровые тела расстреливаются в дробь: медианный объект 6 вокселей
# против 34 после median 3x3x3 при ТОЙ ЖЕ пористости, число компонент 524
# против 250 (diagnostics/check_segmentation.py). Именно из-за этого наши
# первые 3D-картинки выглядели мелкой крошкой рядом с толстыми связными
# телами статьи, где перед Random Walker стоит цепочка фильтров.
air = np.stack([imread(AIR % z).astype(np.float32)[y0:y0 + SUB, x0:x0 + SUB]
                for z in range(z0, z0 + SUB)])
air_dn = ndi.median_filter(air, size=3)
thr_local = float(np.percentile(air_dn, 6.0))
pores = air_dn < thr_local
print(f"куб {SUB}^3 = {SUB*VOXEL_UM:.0f} мкм в ребре")
print(f"  средняя phi {phi.mean():.2f} %, разрешённых пор {100*pores.mean():.2f} % "
      f"(порог {thr_local:.0f}; глобальный порог статьи был {THR_6:.0f})")


def make_grid(scalars, name):
    g = pv.ImageData(dimensions=np.array(scalars.shape) + 1,
                     spacing=(VOXEL_UM,) * 3, origin=(0.0, 0.0, 0.0))
    g.cell_data[name] = scalars.ravel(order="F")
    return g


def setup(plotter, grid):
    """Белый фон, каркас, изометрический вид, параллельная проекция.

    Направление взгляда подобрано под их кадр: видно переднюю грань, верхнюю
    и одну боковую. reset_camera + zoom<1 оставляет поля, иначе куб срезается.
    """
    plotter.set_background("white")
    plotter.add_mesh(grid.outline(), color="black", line_width=1.4)
    plotter.enable_parallel_projection()
    plotter.view_vector((1.0, -1.45, 0.72), viewup=(0, 0, 1))
    plotter.reset_camera()
    plotter.camera.zoom(0.88)


def px_per_um(plotter):
    """При параллельной проекции parallel_scale = полвысоты кадра в мировых единицах."""
    return WIN / (2.0 * plotter.camera.parallel_scale)


# ---------------------------------------------------------------- Fig. 9(a)
smooth = ndi.gaussian_filter(pores.astype(np.float32), 0.9)
gs = make_grid(smooth, "m").cell_data_to_point_data()
surf = gs.contour([0.5], scalars="m").smooth(n_iter=30, relaxation_factor=0.1)
surf.compute_normals(inplace=True, auto_orient_normals=True)
print(f"изоповерхность разрешённых пор: {surf.n_points} вершин")

# Красное облако: субразрешённая пористость. Разрешённые поры исключены -
# иначе они попадут в кадр дважды и забьют синюю поверхность.
sub_phi = phi.copy()
sub_phi[pores] = 0.0
g_sub = make_grid(ndi.gaussian_filter(sub_phi, 1.0), "phi")

p = pv.Plotter(off_screen=True, window_size=(WIN, WIN))
setup(p, g_sub)
p.add_volume(g_sub, scalars="phi", cmap="Reds", clim=[8, 80],
             opacity=[0.0, 0.004, 0.012, 0.03, 0.06, 0.10, 0.16],
             shade=False, show_scalar_bar=False)
p.add_mesh(surf, color="#3a6fd0", smooth_shading=True,
           specular=0.35, specular_power=18, ambient=0.30, diffuse=0.82)
scale_a = px_per_um(p)
p.screenshot("results/render_triple_media.png")
p.close()
print(f"Fig.9(a)-стиль -> results/render_triple_media.png ({scale_a:.3f} px/мкм)")

# ---------------------------------------------------------------- Fig. 9(d)
# Разбор их картинки: у белых тел видны воксельные СТУПЕНЬКИ, а серый фон
# внутри бокса имеет гладкую 2D-структуру. Значит это не чистый volume
# rendering, а грани куба, раскрашенные phi-картой, плюс изоповерхность
# высокой пористости поверх них. Чистый volume rendering здесь и не работает:
# phi=100 в серой шкале - это белый цвет, а на белом фоне он не виден,
# тела читаются только на фоне накопленной серой дымки.
PHI_ISO = 55
g_phi = make_grid(phi, "phi")
faces = g_phi.extract_surface()                       # шесть граней куба
g_sm = make_grid(ndi.gaussian_filter(phi, 0.7), "phi").cell_data_to_point_data()
body = g_sm.contour([PHI_ISO], scalars="phi")
body.compute_normals(inplace=True, auto_orient_normals=True)
print(f"изоповерхность phi>={PHI_ISO}%: {body.n_points} вершин")

p = pv.Plotter(off_screen=True, window_size=(WIN, WIN))
setup(p, g_phi)
# Грани полупрозрачны: у нас phi=0 - это чёрный, и непрозрачные грани дали бы
# сплошной тёмный куб. Белый фон, просвечивая, превращает их в светло-серый
# план, на фоне которого читаются светлые тела - как в статье.
p.add_mesh(faces, scalars="phi", cmap="gray", clim=[0, 100], opacity=0.30,
           lighting=False, show_scalar_bar=True,
           scalar_bar_args=dict(title="Porosity", vertical=True, position_x=0.88,
                                position_y=0.28, width=0.045, height=0.44,
                                color="black", title_font_size=18,
                                label_font_size=14, fmt="%.0f", n_labels=5))
p.add_mesh(body, color="#fbfbfb", smooth_shading=False,
           specular=0.30, specular_power=14, ambient=0.30, diffuse=0.85)
scale_d = px_per_um(p)
p.screenshot("results/render_phi_volume.png")
p.close()
print(f"Fig.9(d)-стиль -> results/render_phi_volume.png ({scale_d:.3f} px/мкм)")


# ------------------------------------------------- сборка сравнения + линейки
def crop_cube(path):
    """Вырезает из панели статьи только сам куб: подписи (a)/(d) и соседние
    панели отделены белым промежутком, поэтому берём самый широкий непрерывный
    блок непустых строк и столбцов."""
    im = plt.imread(path)
    g = im if im.ndim == 2 else im[..., :3].mean(axis=2)
    if g.max() > 1.5:
        g = g / 255.0
    dark = g < 0.82

    def widest(counts, thr):
        idx = np.nonzero(counts > thr)[0]
        if len(idx) == 0:
            return 0, len(counts)
        brk = np.nonzero(np.diff(idx) > 12)[0]
        st = np.r_[idx[0], idx[brk + 1]]
        en = np.r_[idx[brk], idx[-1]]
        k = int(np.argmax(en - st))
        return max(st[k] - 6, 0), min(en[k] + 7, len(counts))

    x0_, x1_ = widest(dark.sum(axis=0), 3)
    y0_, y1_ = widest(dark.sum(axis=1), 3)
    return im[y0_:y1_, x0_:x1_]


def draw_scalebar(ax, img, px_um, um=50, label="50 мкм"):
    """Линейка в стиле статьи: чёрная плашка, белая шкала."""
    h, w = img.shape[:2]
    bar = um * px_um
    bx, by = 0.46 * w, 0.09 * h
    ax.add_patch(Rectangle((bx - 0.10 * bar, by - 0.30 * bar),
                           bar * 1.20, bar * 0.95, fc="black", ec="none", zorder=5))
    ax.add_patch(Rectangle((bx, by), bar, bar * 0.13, fc="white", ec="none", zorder=6))
    ax.text(bx + bar / 2, by + bar * 0.42, label, color="white",
            ha="center", va="center", fontsize=9.5, zorder=7)


fig, ax = plt.subplots(2, 2, figsize=(13, 13.4))
for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])

pairs = [
    (0, "paperfigs/fig9a.png", "results/render_triple_media.png", scale_a,
     "СТАТЬЯ, Fig. 9(a): triple media",
     f"НАШЕ: синее — разрешённые поры (порог статьи, 6 %),\n"
     f"красное — субразрешённая фаза. Куб {SUB}³ = {SUB*VOXEL_UM:.0f} мкм"),
    (1, "paperfigs/fig9d.png", "results/render_phi_volume.png", scale_d,
     "СТАТЬЯ, Fig. 9(d): 3D phi-map",
     "НАШЕ: объёмный рендер phi,\nполная пористость 13.77 %"),
]
for row, ppath, opath, sc, t1, t2 in pairs:
    ax[row, 0].imshow(crop_cube(ppath))
    ax[row, 0].set_title(t1, fontsize=11)
    img = plt.imread(opath)
    ax[row, 1].imshow(img)
    draw_scalebar(ax[row, 1], img, sc)
    ax[row, 1].set_title(t2, fontsize=11)

fig.suptitle("3D-визуализация в стиле статьи", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/fig5_3d_paper_style.png", dpi=95)
print("saved results/fig5_3d_paper_style.png")
