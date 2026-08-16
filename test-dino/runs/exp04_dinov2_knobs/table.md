# Эксперимент 4: перебор ручек DINOv2

Опорная точка — classic-stats из exp02: **0.927**.
Вопрос эксперимента: можно ли настройкой DINOv2 побить простую статистику.

| Модель | Слой | Пулинг | Вход | Скор | Синтетика | Xe |
|---|---|---|---|---|---|---|
| base | L3 | cls | tiles4x224 | mahalanobis | **0.986** | 1.000 |
| base | L3 | cls | tiles4x224 | knn5_cos | **0.985** | 1.000 |
| small | L3 | cls | tiles4x224 | knn5_cos | **0.979** | 1.000 |
| small | L3 | cls | tiles4x224 | mahalanobis | **0.978** | 1.000 |
| small | L6 | cls | tiles4x224 | mahalanobis | **0.969** | 1.000 |
| small | L6 | cls | tiles4x224 | knn5_cos | **0.959** | 1.000 |
| small | L9 | cls | tiles4x224 | knn5_cos | **0.912** | 1.000 |
| small | L9 | cls | tiles4x224 | mahalanobis | **0.909** | 1.000 |
| small | last | cls | tiles4x224 | mahalanobis | **0.833** | 1.000 |
| small | last | cls+mean | tiles4x224 | mahalanobis | **0.830** | 1.000 |
| small | last | mean+std | tiles4x224 | mahalanobis | **0.827** | 1.000 |
| small | last | mean | tiles4x224 | mahalanobis | **0.823** | 1.000 |
| small | last | mean+std | tiles4x224 | knn5_cos | **0.792** | 1.000 |
| small | last | mean | tiles4x224 | knn5_cos | **0.787** | 1.000 |
| small | last | cls+mean | tiles4x224 | knn5_cos | **0.784** | 1.000 |
| small | last | cls | tiles4x224 | knn5_cos | **0.783** | 1.000 |
| small | last | cls+mean | crop224 | mahalanobis | **0.755** | 1.000 |
| small | last | mean+std | crop224 | mahalanobis | **0.754** | 1.000 |
| small | last | mean | crop224 | mahalanobis | **0.754** | 1.000 |
| small | last | cls | crop224 | mahalanobis | **0.752** | 1.000 |

## Итог

- Лучшая конфигурация DINOv2: **0.986** (base/L3/cls/tiles4x224/mahalanobis)
- Классическая статистика: **0.927**
- Разрыв: **+0.059**

## Проверка гипотезы H1 (виновато разрешение)

| Вход | Лучший AUROC |
|---|---|
| tiles4x224 | 0.986 |
| crop224 | 0.755 |
| resize448 | 0.745 |
| resize224 | 0.713 |
| crop448 | 0.710 |

_Время: 2806s_