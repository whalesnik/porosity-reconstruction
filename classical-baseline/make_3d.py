"""3D-модель порового пространства из phi-карты + анализ связности.

Связность - это то, ради чего строится 3D-модель: пористость сама по себе
течение не даёт, флюид идёт только по СВЯЗНОМУ кластеру, протыкающему образец
насквозь. Поэтому здесь для нескольких порогов phi считается:
  - доля объёма,
  - доля, попавшая в наибольший связный кластер,
  - протыкает ли этот кластер образец по z (перколяция).

Работаем на подкубе полного разрешения, а не на прореженном объёме: прореживание
склеивает соседние поры и завышает связность, то есть портит ровно ту величину,
которую мы измеряем.
"""
import json
import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SUB = 300           # ребро подкуба, вокселей (полное разрешение)
VOXEL_UM = 3.0      # из статьи Ebadi et al.: spatial resolution 3 um/vox

cal = json.load(open("results/calibration.json"))
phi_mm = np.load("results/phi_map.npy", mmap_mode="r")
nz, ny, nx = phi_mm.shape

# Связность нельзя считать по прореженному объёму. Если phi-map получена с
# --zstep > 1, воксели по z физически не соседи: при медианной поре в 4 вокселя
# любое прореживание рвёт поровую сеть, и перколяционный анализ выдаст
# уверенный, но неверный ответ. Проверка дешёвая, ошибка - молчаливая.
stats_path = "results/stats.json"
zs = json.load(open(stats_path))["zs"] if os.path.exists(stats_path) else None
zstep = (zs[1] - zs[0]) if zs and len(zs) > 1 else 1
if zstep != 1 or nz != ny or nz != nx:
    raise SystemExit(
        f"phi_map имеет форму {phi_mm.shape} при шаге по z = {zstep}.\n"
        f"Связность и STL считаются только по полному изотропному объёму:\n"
        f"прореживание по z склеивает несоседние срезы и рвёт поровую сеть.\n"
        f"Перезапустите:  python xe_pipeline.py --stage all")
z0, y0, x0 = (nz - SUB) // 2, (ny - SUB) // 2, (nx - SUB) // 2
print(f"полная phi-карта: {phi_mm.shape}, подкуб {SUB}^3 из центра")
phi = np.array(phi_mm[z0:z0 + SUB, y0:y0 + SUB, x0:x0 + SUB])
print(f"средняя пористость подкуба: {phi.mean():.2f} % "
      f"(по всему объёму было {json.load(open('results/stats.json'))['total_porosity']:.2f} %)")

# ---------------------------------------------------------------- связность
STRUCT = ndi.generate_binary_structure(3, 1)     # 6-связность, как в поровой сети
print(f"\n{'порог phi':>10} {'доля об.':>9} {'кластеров':>10} "
      f"{'крупнейший':>11} {'перколяция z':>13}")
print("-" * 60)
conn = []
for T in (50, 35, 25, 15, 10):
    m = phi >= T
    lab, n = ndi.label(m, structure=STRUCT)
    if n == 0:
        continue
    sizes = np.bincount(lab.ravel())[1:]
    big = int(sizes.argmax()) + 1
    frac_big = sizes.max() / m.sum() * 100 if m.sum() else 0
    cluster = lab == big
    perc = bool(cluster[0].any() and cluster[-1].any())
    print(f"{T:>9}% {m.mean()*100:>8.2f}% {n:>10d} {frac_big:>10.1f}% "
          f"{('ДА' if perc else 'нет'):>13}")
    conn.append(dict(threshold=T, vol_frac=float(m.mean() * 100), n_clusters=int(n),
                     largest_frac=float(frac_big), percolates_z=perc))
    del lab, cluster

print("\nЧитается так: разрешённая сеть (высокие пороги) образец НЕ протыкает -")
print("значит проницаемость 0.059 мД из статьи обеспечивается не ей, а")
print("субразрешённой фазой. Модель течения, построенная только по разрешённым")
print("порам, дала бы нулевую проницаемость.")

# ---------------------------------------------------------------- сетка
T_MESH = 35
m = ndi.binary_opening(phi >= T_MESH, STRUCT)     # снимаем одиночные шумовые воксели
sm = ndi.gaussian_filter(m.astype(np.float32), 0.8)
verts, faces, normals, _ = marching_cubes(sm, level=0.5, spacing=(VOXEL_UM,) * 3)
print(f"\nизоповерхность phi={T_MESH}%: {len(verts)} вершин, {len(faces)} граней")

with open("results/pore_network.stl", "wb") as f:
    f.write(b"\0" * 80)
    f.write(struct.pack("<I", len(faces)))
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    for k in range(len(faces)):
        f.write(struct.pack("<12fH", *nrm[k], *tri[k, 0], *tri[k, 1], *tri[k, 2], 0))
mb = (84 + 50 * len(faces)) / 1e6
print(f"сохранено results/pore_network.stl ({mb:.1f} МБ) - "
      f"открывается в ParaView / Blender / MeshLab")

# ---------------------------------------------------------------- картинки
fig = plt.figure(figsize=(18, 11))

# Для обзорной картинки берём куб поменьше: 4 млн граней с подкуба 300^3
# matplotlib рисует сплошным комком, в котором ничего не разобрать.
# Полная сетка уходит в STL - её смотреть в ParaView.
VIS = 128
v0 = (SUB - VIS) // 2
mv = ndi.binary_opening(phi[v0:v0 + VIS, v0:v0 + VIS, v0:v0 + VIS] >= T_MESH, STRUCT)
sv = ndi.gaussian_filter(mv.astype(np.float32), 0.8)
vv, ff, _, _ = marching_cubes(sv, level=0.5, spacing=(VOXEL_UM,) * 3)
ax = fig.add_subplot(2, 3, 1, projection="3d")
ax.plot_trisurf(vv[:, 0], vv[:, 1], ff, vv[:, 2], cmap="copper", lw=0, antialiased=False)
ax.set_title(f"3D поровое пространство, phi>={T_MESH}%\n"
             f"обзорный куб {VIS}^3 ({VIS*VOXEL_UM:.0f} мкм в ребре)")
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=22, azim=35)

for k, (idx, lbl) in enumerate([(SUB // 2, "XY"), (SUB // 2, "XZ"), (SUB // 2, "YZ")]):
    a = fig.add_subplot(2, 3, k + 2)
    img = [phi[idx], phi[:, idx], phi[:, :, idx]][k]
    im = a.imshow(img, cmap="viridis", vmin=0, vmax=100)
    a.set_title(f"phi-map, сечение {lbl}"); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=a, fraction=.046, label="phi, %")

a = fig.add_subplot(2, 3, 5)
Ts = [c["threshold"] for c in conn]
a.plot(Ts, [c["vol_frac"] for c in conn], "o-", label="доля объёма")
a.plot(Ts, [c["largest_frac"] for c in conn], "s-", label="в крупнейшем кластере, %")
for c in conn:
    if c["percolates_z"]:
        a.axvline(c["threshold"], c="g", alpha=.25, lw=6)
a.set_xlabel("порог phi, %"); a.legend(fontsize=9); a.grid(alpha=.3)
a.invert_xaxis()
a.set_title("связность поровой сети\n(зелёная полоса = сеть протыкает образец)")

a = fig.add_subplot(2, 3, 6)
prof = phi.reshape(SUB, -1).mean(axis=1)
a.plot(prof, lw=1.3)
a.axhline(phi.mean(), c="r", ls="--", label=f"среднее {phi.mean():.2f} %")
a.set_xlabel("z подкуба"); a.set_ylabel("phi, %"); a.legend(fontsize=9); a.grid(alpha=.3)
a.set_title("пористость по z внутри подкуба")

plt.tight_layout(); plt.savefig("results/fig3_3d.png", dpi=95)
json.dump(conn, open("results/connectivity.json", "w"), indent=2)
print("saved results/fig3_3d.png, results/connectivity.json")
