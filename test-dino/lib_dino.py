"""
Общая библиотека для экспериментов test-dino (Человек 5, OOD-детектор).

Здесь всё переиспользуемое: загрузка срезов, варианты препроцессинга,
извлечение эмбеддингов разными бэкбонами, OOD-скоры и метрики.
Сами эксперименты — в exp*.py, каждый пишет в runs/<имя>/.
"""

from __future__ import annotations

import os
from pathlib import Path

TEST_DINO = Path(__file__).resolve().parent
MODELS_DIR = TEST_DINO / "models"
# torchvision качает веса в глобальный кэш ~/.cache/torch — уводим их в папку проекта,
# чтобы все веса лежали вместе с экспериментом, а не в пользовательском кэше.
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = TEST_DINO.parent
DATA_ROOT = PROJECT_ROOT / "hgxdh8ps94-1"
AIR_DIR = DATA_ROOT / "Air"
XE_DIR = DATA_ROOT / "Xenon_"
RUNS_DIR = TEST_DINO / "runs"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- загрузка ---

def list_slice_files(dir_path: Path, prefix: str) -> list[Path]:
    return sorted(dir_path.glob(f"{prefix}_*.png"), key=lambda p: int(p.stem.split("_")[-1]))


def load_slices(dir_path: Path, prefix: str, n_slices: int, z_range: tuple[float, float] = (0.0, 1.0)) -> tuple[np.ndarray, list[int]]:
    """n_slices равномерно распределённых срезов из заданного диапазона по глубине.
    z_range задаётся долями (0.0, 1.0) = весь стек; (0.0, 0.5) = верхняя половина."""
    files = list_slice_files(dir_path, prefix)
    lo = int(z_range[0] * (len(files) - 1))
    hi = int(z_range[1] * (len(files) - 1))
    idxs = np.linspace(lo, hi, n_slices, dtype=int)
    arrs = [np.array(Image.open(files[i])).astype(np.float32) for i in idxs]
    return np.stack(arrs), [int(i) for i in idxs]


def load_air(n: int, z_range=(0.0, 1.0)):
    return load_slices(AIR_DIR, "Air", n, z_range)


def load_xe(n: int, z_range=(0.0, 1.0)):
    return load_slices(XE_DIR, "Xe", n, z_range)


# ---------------------------------------------------------- нормализации ---
# Каждая функция: (H,W) float32 -> (H,W) float32 примерно в [0,1].
# global_stats нужен тем вариантам, которые нормируют по общей шкале обоих сканов,
# а не по каждому срезу отдельно.

def norm_minmax_per_slice(img: np.ndarray, **kw) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def norm_percentile_per_slice(img: np.ndarray, p_lo=1.0, p_hi=99.0, **kw) -> np.ndarray:
    """Устойчивее min-max: игнорирует одиночные выбросы яркости (артефакты сканера)."""
    lo, hi = np.percentile(img, p_lo), np.percentile(img, p_hi)
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(img)


def norm_global_window(img: np.ndarray, global_lo: float = None, global_hi: float = None, **kw) -> np.ndarray:
    """ОДНО общее окно яркости для всех срезов обоих сканов. Убирает разницу
    в общей яркости между Air и Xe как артефакт шкалы — остаётся только структура."""
    return np.clip((img - global_lo) / (global_hi - global_lo), 0, 1)


def norm_zscore_per_slice(img: np.ndarray, **kw) -> np.ndarray:
    z = (img - img.mean()) / (img.std() + 1e-8)
    return np.clip(z * 0.2 + 0.5, 0, 1)  # z -> примерно [0,1]


def norm_clahe(img: np.ndarray, **kw) -> np.ndarray:
    """Локальное выравнивание контраста: подчёркивает текстуру, гасит глобальный
    яркостный сдвиг. Самый агрессивный способ убрать 'просто разную яркость'."""
    from skimage.exposure import equalize_adapthist
    x = norm_percentile_per_slice(img)
    return equalize_adapthist(x, clip_limit=0.01).astype(np.float32)


def norm_hist_eq(img: np.ndarray, **kw) -> np.ndarray:
    from skimage.exposure import equalize_hist
    return equalize_hist(img).astype(np.float32)


def norm_identity(img: np.ndarray, **kw) -> np.ndarray:
    """Ничего не делать — данные уже в общей физической шкале.
    Максимальная чувствительность к сдвигу яркости."""
    return np.clip(img, 0, 1)


NORMALIZATIONS = {
    "minmax": norm_minmax_per_slice,
    "percentile": norm_percentile_per_slice,
    "global_window": norm_global_window,
    "zscore": norm_zscore_per_slice,
    "clahe": norm_clahe,
    "hist_eq": norm_hist_eq,
}


def compute_global_window(volumes: list[np.ndarray], p_lo=0.5, p_hi=99.5) -> tuple[float, float]:
    sample = np.concatenate([v[::max(1, len(v) // 10)].ravel()[::37] for v in volumes])
    return float(np.percentile(sample, p_lo)), float(np.percentile(sample, p_hi))


# ------------------------------------------------------------ маска керна ---

def core_mask_from_slice(img: np.ndarray) -> np.ndarray:
    """ВНИМАНИЕ: применимость зависит от датасета (проверено в exp00 и exp06).

    Сколтех (hgxdh8ps94-1): фона и держателя НЕТ — 900x900 целиком заполнены
    породой, это уже вырезанный внутренний субобъём. Порог Отсу здесь выделяет
    не 'керн против фона', а яркие минеральные зёрна (~15% пикселей).
    Маска на этих данных не нужна и вредна, нужен center_box.

    Свой датасет (22322-24): фон и держатель ЕСТЬ — круглый керн на чёрном фоне.
    Здесь маска обязательна, иначе признаки будут описывать фон, а не породу."""
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes, binary_opening
    thr = threshold_otsu(img)
    m = img > thr
    m = binary_opening(m, np.ones((5, 5)))
    return binary_fill_holes(m)


def inscribed_box_from_mask(mask: np.ndarray, margin: float = 0.80) -> tuple[int, int, int]:
    """Квадрат, гарантированно лежащий внутри круглого керна.
    Радиус берётся с запасом (margin), чтобы не зацепить край и держатель."""
    from scipy.ndimage import label, binary_erosion
    lab, n = label(mask)
    if n > 1:  # оставляем только самую большую компоненту — сам керн
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == sizes.argmax()
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt(mask.sum() / np.pi) * margin
    half = int(r / np.sqrt(2))
    return int(cy - half), int(cx - half), 2 * half


def center_box(shape: tuple[int, int], frac: float = 0.9) -> tuple[int, int, int]:
    """Центральный квадрат кадра. Замена largest_inscribed_square: раз фона нет,
    достаточно просто взять центр."""
    h, w = shape
    size = int(min(h, w) * frac)
    return (h - size) // 2, (w - size) // 2, size


# ------------------------------------------------- подготовка входа модели ---

def prepare_2d(img01: np.ndarray, target_size: int = 224, mode: str = "resize",
               crop_box: tuple[int, int, int] | None = None) -> np.ndarray:
    """
    img01: (H,W) в [0,1] -> (S,S) в [0,1], без ImageNet-нормализации.
    Отдельно от to_tensor, чтобы классические признаки (гистограмма, GLCM)
    считались ровно на том же изображении, что видит нейросеть.

    mode='resize' — сжать весь срез 900->224 (теряется мелкая текстура пор:
        4-кратное уменьшение съедает как раз тот масштаб, который нас интересует).
    mode='crop'   — вырезать 224x224 в нативном разрешении из центра керна
        (текстура сохраняется, но виден только фрагмент).
    """
    if mode == "crop":
        r0, c0, size = crop_box
        cy, cx = r0 + size // 2, c0 + size // 2
        h = target_size // 2
        return img01[cy - h: cy + h, cx - h: cx + h]
    x = torch.from_numpy(img01).float()[None, None]
    x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return x[0, 0].numpy()


def to_tensor(img2d: np.ndarray) -> torch.Tensor:
    """(S,S) в [0,1] -> (1,3,S,S), ImageNet-нормализованный."""
    x = torch.from_numpy(np.ascontiguousarray(img2d)).float()[None, None].repeat(1, 3, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def to_model_input(img01: np.ndarray, target_size: int = 224, mode: str = "resize",
                   crop_box: tuple[int, int, int] | None = None) -> torch.Tensor:
    return to_tensor(prepare_2d(img01, target_size, mode, crop_box))


def tile_crops(img01: np.ndarray, tile: int = 224, n_tiles: int = 4, rng=None) -> list[np.ndarray]:
    """Несколько 224x224 тайлов в нативном разрешении из центральной области кадра."""
    rng = rng or np.random.default_rng(0)
    r0, c0, size = center_box(img01.shape)
    tiles = []
    for _ in range(n_tiles):
        rr = rng.integers(r0, r0 + size - tile)
        cc = rng.integers(c0, c0 + size - tile)
        tiles.append(img01[rr:rr + tile, cc:cc + tile])
    return tiles


# ------------------------------------------------------------- бэкбоны ---

class Backbone:
    """Единый интерфейс: (1,3,H,W) -> (1,D) эмбеддинг."""

    def __init__(self, name: str, model, kind: str, pooling: str = "cls",
                 layer: int | None = None):
        self.name, self.model, self.kind, self.pooling = name, model, kind, pooling
        # layer=None — последний слой; иначе индекс скрытого состояния (0 = вход эмбеддингов).
        # Для текстур средние слои часто информативнее финального, заточенного под семантику.
        self.layer = layer

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "hf_vit":
            if self.layer is None:
                out = self.model(pixel_values=x).last_hidden_state
            else:
                hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
                out = hs[self.layer]
            if self.pooling == "cls":
                return out[:, 0, :]
            if self.pooling == "mean":
                return out[:, 1:, :].mean(dim=1)
            if self.pooling == "cls+mean":
                return torch.cat([out[:, 0, :], out[:, 1:, :].mean(dim=1)], dim=1)
            if self.pooling == "mean+std":
                p = out[:, 1:, :]
                return torch.cat([p.mean(dim=1), p.std(dim=1)], dim=1)
        if self.kind == "timm_cnn":
            return self.model(x)
        raise ValueError(self.kind)


def load_dinov2(size: str = "base", pooling: str = "cls", local_only: bool = True,
                layer: int | None = None) -> Backbone:
    from transformers import AutoModel
    local_path = MODELS_DIR / f"dinov2-{size}"
    src = str(local_path) if local_path.exists() else f"facebook/dinov2-{size}"
    model = AutoModel.from_pretrained(src)
    model.eval()
    tag = f"dinov2-{size}[{pooling}]" + ("" if layer is None else f"@L{layer}")
    return Backbone(tag, model, "hf_vit", pooling, layer)


def load_resnet50() -> Backbone:
    """Классический supervised-ImageNet бэкбон — контрольная точка: даёт ли
    самообучение DINOv2 что-то сверх обычных ImageNet-признаков."""
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m.eval()
    return Backbone("resnet50-imagenet", m, "timm_cnn")


@torch.no_grad()
def embed_batch(backbone: Backbone, inputs: list[torch.Tensor], batch_size: int = 8) -> torch.Tensor:
    outs = []
    for i in range(0, len(inputs), batch_size):
        batch = torch.cat(inputs[i:i + batch_size], dim=0)
        outs.append(backbone(batch))
    return torch.cat(outs, dim=0)


# ---------------------------------------------- классические baseline-фичи ---

def features_histogram(img01: np.ndarray, bins: int = 64) -> np.ndarray:
    h, _ = np.histogram(img01, bins=bins, range=(0, 1), density=True)
    return h.astype(np.float32)


def features_glcm(img01: np.ndarray) -> np.ndarray:
    """Haralick-текстурные признаки (GLCM). Классика анализа текстур до эры DL —
    нужна как честный baseline: если она не хуже DINOv2, нейросеть тут не нужна."""
    from skimage.feature import graycomatrix, graycoprops
    q = (np.clip(img01, 0, 1) * 63).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props]).astype(np.float32)


def features_basic_stats(img01: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    gx = ndimage.sobel(img01, axis=0)
    gy = ndimage.sobel(img01, axis=1)
    grad = np.hypot(gx, gy)
    return np.array([
        img01.mean(), img01.std(),
        *np.percentile(img01, [1, 5, 25, 50, 75, 95, 99]),
        grad.mean(), grad.std(),
    ], dtype=np.float32)


# --------------------------------------------------------------- OOD-скоры ---

def knn_score(query: torch.Tensor, reference: torch.Tensor, k: int = 5, metric: str = "cosine") -> torch.Tensor:
    """Среднее расстояние до k ближайших соседей в reference-множестве.
    Стандартный непараметрический OOD-скор (Sun et al., 2022)."""
    if metric == "cosine":
        q = F.normalize(query, dim=1)
        r = F.normalize(reference, dim=1)
        d = 1 - q @ r.T
    else:
        d = torch.cdist(query, reference)
    k = min(k, d.shape[1])
    return d.topk(k, largest=False).values.mean(dim=1)


def mahalanobis_score(query: torch.Tensor, reference: torch.Tensor, shrinkage: float = 0.1) -> torch.Tensor:
    """Расстояние Махаланобиса до распределения reference. Учитывает ковариацию
    признаков; shrinkage нужен, т.к. образцов меньше, чем размерность."""
    mu = reference.mean(dim=0, keepdim=True)
    xc = (reference - mu).double()
    cov = (xc.T @ xc) / max(1, len(reference) - 1)
    cov = (1 - shrinkage) * cov + shrinkage * torch.eye(cov.shape[0], dtype=torch.float64) * cov.diagonal().mean()
    prec = torch.linalg.pinv(cov)
    d = (query - mu).double()
    return torch.sqrt(torch.clamp((d @ prec * d).sum(dim=1), min=0)).float()


def auroc(scores_in: torch.Tensor, scores_out: torch.Tensor) -> float:
    """Насколько хорошо скор отделяет OOD от in-distribution. 0.5 = случайно, 1.0 = идеально."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(scores_in)), np.ones(len(scores_out))]
    s = np.r_[scores_in.numpy(), scores_out.numpy()]
    return float(roc_auc_score(y, s))


def linear_probe_auc(emb_a: torch.Tensor, emb_b: torch.Tensor, n_splits: int = 5) -> float:
    """Может ли простой линейный классификатор отделить две группы по эмбеддингам.
    Честнее t-SNE: t-SNE может 'нарисовать' разделение там, где его почти нет.

    ВАЖНО: при размерности признаков >> числа образцов (768 против ~80) этот
    показатель почти всегда близок к 1.0 даже для случайных групп. Использовать
    только вместе с probe_control_auc в качестве точки отсчёта."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = np.r_[emb_a.numpy(), emb_b.numpy()]
    y = np.r_[np.zeros(len(emb_a)), np.ones(len(emb_b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=min(n_splits, len(y) // 2), shuffle=True, random_state=0)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def probe_control_auc(emb_a: torch.Tensor, n_repeats: int = 3, seed: int = 0) -> float:
    """ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ: делим ОДНУ однородную группу пополам случайно и
    прогоняем тот же самый пробник. Настоящего различия между половинами нет,
    поэтому честная метрика обязана дать ~0.5. Если получается ~1.0 — значит
    метрика меряет переобучение в многомерном пространстве, а не реальный сигнал,
    и все выводы 'AUC=1.0, группы различаются' недействительны."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_repeats):
        perm = rng.permutation(len(emb_a))
        half = len(perm) // 2
        aucs.append(linear_probe_auc(emb_a[perm[:half]], emb_a[perm[half:2 * half]]))
    return float(np.mean(aucs))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, gamma: float | None = None) -> float:
    """Maximum Mean Discrepancy — расстояние между РАСПРЕДЕЛЕНИЯМИ, а не между
    отдельными точками. В отличие от линейного пробника не переобучается на
    размерности, поэтому годится там, где probe_auc бесполезен."""
    X = torch.cat([x, y]).double()
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    a, b = X[:len(x)], X[len(x):]
    d2 = torch.cdist(X, X) ** 2
    if gamma is None:
        med = d2[d2 > 0].median()
        gamma = 1.0 / (med + 1e-12)
    K = torch.exp(-gamma * d2)
    n, m = len(a), len(b)
    Kxx = (K[:n, :n].sum() - K[:n, :n].diagonal().sum()) / (n * (n - 1))
    Kyy = (K[n:, n:].sum() - K[n:, n:].diagonal().sum()) / (m * (m - 1))
    Kxy = K[:n, n:].mean()
    return float(torch.clamp(Kxx + Kyy - 2 * Kxy, min=0).sqrt())


# ------------------------------------------------------------- искажения ---
# Контролируемый домен-сдвиг: своего второго датасета сегодня нет, поэтому
# имитируем "другое месторождение/другой сканер" известными искажениями.

def corrupt_brightness(img01, severity):
    return np.clip(img01 + [0.02, 0.05, 0.10, 0.20, 0.35][severity - 1], 0, 1)


def corrupt_contrast(img01, severity):
    f = [0.9, 0.75, 0.6, 0.45, 0.3][severity - 1]
    return np.clip((img01 - img01.mean()) * f + img01.mean(), 0, 1)


def corrupt_noise(img01, severity, rng=None):
    rng = rng or np.random.default_rng(0)
    s = [0.01, 0.02, 0.04, 0.08, 0.15][severity - 1]
    return np.clip(img01 + rng.normal(0, s, img01.shape), 0, 1).astype(np.float32)


def corrupt_blur(img01, severity):
    from scipy.ndimage import gaussian_filter
    s = [0.5, 1.0, 2.0, 3.5, 6.0][severity - 1]
    return gaussian_filter(img01, s)


def corrupt_resolution(img01, severity):
    """Имитация худшего разрешения сканера — самый физически осмысленный
    сдвиг для нашей задачи (субразрешённые поры именно про это)."""
    f = [1.5, 2, 3, 4, 6][severity - 1]
    x = torch.from_numpy(img01).float()[None, None]
    small = F.interpolate(x, scale_factor=1 / f, mode="area")
    back = F.interpolate(small, size=img01.shape, mode="bilinear", align_corners=False)
    return back[0, 0].numpy()


CORRUPTIONS = {
    "brightness": corrupt_brightness,
    "contrast": corrupt_contrast,
    "noise": corrupt_noise,
    "blur": corrupt_blur,
    "resolution": corrupt_resolution,
}


# ------------------------- структурные сдвиги (другая порода, а не другой сканер) ---
# Отличие от CORRUPTIONS: здесь меняется геометрия порового пространства, а не
# параметры съёмки. Детектор доверия обязан их ловить — это и есть "незнакомая порода".

def _pore_mask(img01: np.ndarray) -> np.ndarray:
    """Поры = тёмная фаза. Порог Отсу по самому срезу."""
    from skimage.filters import threshold_otsu
    return img01 < threshold_otsu(img01)


def structural_pore_dilate(img01: np.ndarray, severity: int) -> np.ndarray:
    """Поры крупнее при той же минералогии — физически это другой тип породы
    (лучше отсортированный песчаник), а не другая настройка сканера."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    r = [1, 2, 3, 4, 6][severity - 1]
    pores = _pore_mask(img01)
    grown = binary_dilation(pores, iterations=r)
    dark = float(np.percentile(img01[pores], 50)) if pores.any() else 0.0
    out = np.where(grown & ~pores, dark, img01)
    return gaussian_filter(out.astype(np.float32), 0.5)


def structural_pore_erode(img01: np.ndarray, severity: int) -> np.ndarray:
    """Поры мельче — имитация более плотной, хуже отсортированной породы."""
    from scipy.ndimage import binary_erosion, gaussian_filter
    r = [1, 2, 3, 4, 6][severity - 1]
    pores = _pore_mask(img01)
    shrunk = binary_erosion(pores, iterations=r)
    bright = float(np.percentile(img01[~pores], 50)) if (~pores).any() else 1.0
    out = np.where(pores & ~shrunk, bright, img01)
    return gaussian_filter(out.astype(np.float32), 0.5)


def structural_elastic(img01: np.ndarray, severity: int, rng=None) -> np.ndarray:
    """Упругая деформация: геометрия пор искажена, гистограмма почти не тронута."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    rng = rng or np.random.default_rng(0)
    amp = [2, 4, 8, 14, 22][severity - 1]
    h, w = img01.shape
    dx = gaussian_filter(rng.normal(0, 1, (h, w)), 12) * amp
    dy = gaussian_filter(rng.normal(0, 1, (h, w)), 12) * amp
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)])
    return map_coordinates(img01, coords, order=1, mode="reflect").astype(np.float32)


def structural_block_shuffle(img01: np.ndarray, severity: int, rng=None) -> np.ndarray:
    """Перестановка блоков: гистограмма сохраняется ТОЧНО, пространственная
    связность разрушена. Самый чистый тест на чувствительность к структуре —
    любой признак, основанный только на распределении яркостей, здесь слеп."""
    rng = rng or np.random.default_rng(0)
    b = [112, 56, 28, 16, 8][severity - 1]
    h, w = img01.shape
    nh, nw = h // b, w // b
    blocks = (img01[:nh * b, :nw * b]
              .reshape(nh, b, nw, b).transpose(0, 2, 1, 3).reshape(nh * nw, b, b))
    blocks = blocks[rng.permutation(len(blocks))]
    out = blocks.reshape(nh, nw, b, b).transpose(0, 2, 1, 3).reshape(nh * b, nw * b)
    res = img01.copy()
    res[:nh * b, :nw * b] = out
    return res


STRUCTURAL = {
    "pore_dilate": structural_pore_dilate,
    "pore_erode": structural_pore_erode,
    "elastic": structural_elastic,
    "block_shuffle": structural_block_shuffle,
}

# Помехи съёмки: детектор доверия должен их ИГНОРИРОВАТЬ (другой сканер — та же порода).
NUISANCE = {
    "brightness": corrupt_brightness,
    "contrast": corrupt_contrast,
    "noise": corrupt_noise,
}
