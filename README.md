# G-MAE: Маскированный автокодировщик для жестов

Этот репозиторий содержит реализацию модели G-MAE (Gesture-aware Masked Autoencoder) на PyTorch для задач распознавания жестов.

Проект включает этап предобучения модели на неразмеченных изображениях жестов и дообучения (fine-tuning) для задачи классификации на размеченных данных.

## Оглавление

1.  [Введение](#1-введение)
2.  [Обзор архитектуры](#2-обзор-архитектуры)
3.  [Структура проекта](#3-структура-проекта)
4.  [Установка и запуск](#4-установка-и-запуск)
5.  [Подготовка данных](#5-подготовка-данных)
6.  [Обучение](#6-обучение)
    *   [Предобучение](#предобучение)
    *   [Дообучение](#дообучение)
7.  [Воспроизводимость](#7-воспроизводимость)

## 1. Введение

G-MAE — это метод self-supervised обучения для распознавания жестов. Архитектура основана на модифицированном трансформере (GMST), который сочетает сверточные и attention-механизмы для захвата как локальных, так и глобальных зависимостей. Модель адаптирована для работы с изображениями жестов.

## 2. Обзор архитектуры

Модель G-MAE состоит из энкодера и декодера и работает в два этапа:

### 2.1. Модель G-MAE

*   **Энкодер**: обрабатывает входные изображения (или видимые части замаскированных изображений) и извлекает латентное представление.
*   **Декодер**: восстанавливает исходное изображение из латентного представления энкодера (используется только на этапе предобучения).
*   **Маскирование**: на этапе предобучения случайным образом маскируется значительная часть изображения (например, 60%). Модель учится восстанавливать замаскированные области.
*   **Функции потерь**: MSE для предобучения (реконструкция), Cross-Entropy для дообучения (классификация).

### 2.2. GMST Backbone

Основой G-MAE является модуль GMST (Gesture-aware Multi-Scale Transformer), который включает:

*   **OverlapPatchEmbed**: преобразует изображение в последовательность перекрывающихся патчей и эмбеддит их.
*   **GMSTBlock**: основной строительный блок энкодера и декодера. Включает:
    *   **MSDC**: многомасштабные дилатированные свертки для захвата пространственных паттернов.
    *   **MHSA**: multi-head self-attention для глобальных зависимостей.
    *   **MSC-FFN**: многомасштабная feed-forward сеть.

В модели используются остаточные связи и layer normalization для стабильного обучения.

## 3. Структура проекта

```
gmae_project/
├── src/
│   ├── configs/                # Конфигурации
│   │   ├── __init__.py
│   │   ├── base_config.py      # Базовые параметры
│   │   ├── pretrain.yaml       # Конфиг предобучения
│   │   ├── finetune.yaml       # Конфиг дообучения
│   │   └── inference.yaml      # Конфиг инференса
│   │
│   ├── data/                   # Датасеты и преобразования
│   │   ├── __init__.py
│   │   ├── datasets.py         # Кастомные Dataset-классы
│   │   ├── transforms.py       # Аугментации и нормализации
│   │   ├── data_formater.py    # Форматирование и разбиение по папкам
│   │   ├── prepare_corpus.py   # Фильтрация/очистка корпуса
│   │   └── split_corpus.py     # Разбиение на train/val/test
│   │
│   ├── engine/                 # Обёртки для цикла обучения
│   │   ├── __init__.py
│   │   └── trainer.py          # Trainer (fit/evaluate)
│   │
│   ├── models/                 # Описание архитектуры
│   │   ├── __init__.py
│   │   ├── gmae.py             # Основной класс GMAE
│   │   └── modules.py          # MSDC, MHSA, MSC-FFN, GMSTBlock, OverlapPatchEmbed
│   │
│   ├── scripts/                # CLI-скрипты запуска
│   │   ├── __init__.py
│   │   ├── pretrain.py         # Запуск предобучения
│   │   ├── finetune.py         # Запуск дообучения
│   │   └── inference.py        # Запуск инференса/оценки
│   │
│   └── utils/                  # Вспомогательные утилиты
│       ├── __init__.py
│       ├── helpers.py          # Частые функции (метрики, и т.п.)
│       └── logger.py           # Логирование и отслеживание прогресса
│
├── train_pretrain.sh           # Быстрый запуск предобучения
├── train_finetune.sh           # Быстрый запуск дообучения
├── README.md                   # Описание проекта
└── pyproject.toml              # Зависимости (Poetry/PEP-621)
```

## 4. Установка и запуск

### 4.1. Требования

*   **Python**: версия 3.8+
*   **CUDA**: для ускорения на GPU (опционально)
*   **Docker**: для воспроизводимости (опционально)

### 4.2. Docker (рекомендуется для воспроизводимости)

1.  **Сборка Docker-образа**:
    ```bash
    docker build -t gmae-training .
    ```
2.  **Запуск контейнера**:
    ```bash
    docker run --gpus all -it --rm -v /path/to/your/local/data:/app/data gmae-training bash
    ```
    Замените `/path/to/your/local/data` на абсолютный путь к вашей папке с данными.

### 4.3. Локальная установка (альтернатива)

1.  **Клонирование репозитория**:
    ```bash
    git clone <адрес_репозитория> gmae_project
    cd gmae_project
    ```
2.  **Создание виртуального окружения**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Установка зависимостей**:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # или для CUDA
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install tqdm Pillow numpy
    ```

## 5. Подготовка данных

Для обучения использовался датасет Hagrid. Для этапа предобучения использовалось 70% данных, для этапа дообучения — оставшиеся 30%. Данные были разбиты на train/val с помощью скриптов из папки `src/data/`.

**Структура данных:**
```
data/
├── pretrain_dataset/
│   ├── train/
│   ├── val/
└── finetune_dataset/
    ├── train/
    ├── val/
```

Каждая папка содержит подпапки с названиями классов, внутри которых лежат изображения.

### 5.1. Аугментации и преобразования

В файле `src/data/transforms.py` реализованы аугментации для train/val:

*   **Для обучения**:
    *   Случайный кроп и ресайз с сохранением пропорций
    *   Случайное горизонтальное отражение
    *   Случайные изменения яркости, контраста, насыщенности, оттенка
    *   Нормализация по ImageNet
*   **Для валидации/теста**:
    *   Ресайз и центр-кроп
    *   Нормализация по ImageNet

## 6. Обучение

Обучение состоит из двух этапов: предобучение и дообучение. Для запуска используются bash-скрипты.

### Предобучение

Модель обучается как маскированный автокодировщик на неразмеченных изображениях жестов (70% Hagrid).

Запуск:
```bash
./train_pretrain.sh
```

**Основные параметры (можно менять в train_pretrain.sh или src/scripts/pretrain.py):**
*   `--epochs`: количество эпох (по умолчанию 500)
*   `--batch_size`: размер батча (по умолчанию 64)
*   `--learning_rate`: начальный learning rate (по умолчанию 1e-3)
*   `--mask_ratio`: доля маскируемых патчей (по умолчанию 0.6)
*   и др.

### Дообучение

На этом этапе энкодер дообучается для задачи классификации жестов (30% Hagrid, размеченные данные).

Запуск:
```bash
./train_finetune.sh
```

**Основные параметры (можно менять в train_finetune.sh или src/scripts/finetune.py):**
*   `--epochs`: количество эпох (по умолчанию 250)
*   `--batch_size`: размер батча (по умолчанию 64)
*   `--learning_rate`: начальный learning rate (по умолчанию 1e-4)
*   `--num_classes`: количество классов
*   и др.

## 7. Воспроизводимость

Для воспроизводимости рекомендуется использовать Docker. В скриптах обучения фиксируются random seed'ы.
