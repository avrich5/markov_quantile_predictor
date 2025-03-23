# Markov Quantile Predictor

Гибридная модель для предсказания временных рядов, объединяющая марковские цепи и квантильную регрессию.

## Описание проекта

Markov Quantile Predictor представляет собой гибридный подход к прогнозированию временных рядов. Проект включает три основные модели:

1. **MarkovPredictor** - базовая модель на основе марковских цепей
2. **MarkovQuantilePredictor** - простая гибридная модель, объединяющая марковские цепи и квантильную регрессию
3. **EnhancedHybridPredictor** - улучшенная гибридная модель с расширенными характеристиками состояний

Такой подход позволяет не только предсказывать направление движения цены, но и оценивать вероятностные интервалы будущих значений.

## Структура проекта

```
markov_quantile_predictor/
├── __init__.py
├── predictor_config.py
├── factory.py
├── utils.py
├── config/
│   ├── __init__.py
│   ├── defaults.py
│   └── presets.py
├── models/
│   ├── __init__.py
│   ├── markov_predictor.py       # Базовая марковская модель
│   ├── hybrid_predictor.py       # Гибридные модели (MarkovQuantilePredictor и EnhancedHybridPredictor)
│   ├── quantile_regression.py    # Модуль квантильной регрессии
│   └── early_stopping.py         # Функциональность раннего останова
└── examples/
    ├── simple_example.py             # Простой пример использования
    ├── new_config_usage.py           # Пример работы с конфигурациями
    ├── validate_enhanced_hybrid.py   # Валидация улучшенной гибридной модели
    ├── test_enhanced_hybrid.py       # Тестирование улучшенной гибридной модели
    └── compare_models.py             # Сравнение различных моделей
```

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/markov_quantile_predictor.git
cd markov_quantile_predictor

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в режиме разработки
pip install -e .
```

## Модели прогнозирования

### MarkovPredictor

Базовая модель, анализирующая последовательности состояний рынка с помощью марковских цепей. Используется для определения вероятностей перехода между различными состояниями рынка.

```python
from markov_quantile_predictor import create_predictor

# Создание марковского предиктора
predictor = create_predictor(model_type="markov")
results = predictor.run_on_data(prices)
```

### MarkovQuantilePredictor

Простая гибридная модель, которая объединяет марковский предиктор с квантильной регрессией. Определяет состояние рынка с помощью марковских цепей, затем применяет квантильную регрессию для прогнозирования распределения будущих значений.

```python
from markov_quantile_predictor import create_predictor

# Создание простого гибридного предиктора
predictor = create_predictor(model_type="hybrid")
results = predictor.run_on_data(prices)
```

### EnhancedHybridPredictor

Улучшенная гибридная модель, использующая расширенные характеристики состояний рынка. Включает дополнительные факторы и улучшенные алгоритмы взвешивания для повышения точности прогнозов.

```python
from markov_quantile_predictor import create_predictor

# Создание улучшенного гибридного предиктора
predictor = create_predictor(model_type="enhanced_hybrid")
results = predictor.run_on_data(prices)
```

## Настройка конфигурации

### Конфигурация предиктора

Класс `PredictorConfig` позволяет настроить все параметры моделей:

```python
from markov_quantile_predictor import PredictorConfig

config = PredictorConfig(
    # Общие параметры
    window_size=750,               # Размер окна (количество точек для обучения)
    prediction_depth=15,           # Глубина предсказания (на сколько точек вперед)
    min_confidence=0.6,            # Минимальная уверенность для предсказания
    
    # Параметры марковского предиктора
    state_length=4,                # Длина последовательности состояний
    significant_change_pct=0.4,    # Порог значимого изменения в %
    use_weighted_window=False,     # Использовать ли взвешенное окно
    
    # Параметры квантильной регрессии
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95), # Квантили для регрессии
    min_samples_for_regression=10, # Минимум образцов для регрессии
    
    # Параметры фильтрации предсказаний
    confidence_threshold=0.5,      # Порог уверенности для фильтрации
    max_coverage=0.05              # Максимальное покрытие (доля предсказаний)
)
```

### Предустановки конфигураций

Для удобства в пакете доступны предустановленные конфигурации для различных сценариев:

| Предустановка | Описание |
|---------------|----------|
| `standard` | Стандартная сбалансированная конфигурация |
| `high_precision` | Конфигурация для повышенной точности предсказаний |
| `high_coverage` | Конфигурация для максимального покрытия данных |
| `high_volatility` | Конфигурация для рынков с высокой волатильностью |
| `low_volatility` | Конфигурация для рынков с низкой волатильностью |
| `quick_test` | Конфигурация для быстрого тестирования |

```python
from markov_quantile_predictor import create_predictor

# Создание предиктора с предустановкой
predictor = create_predictor(model_type="enhanced_hybrid", preset_name="high_volatility")
```

### Переопределение параметров предустановки

```python
from markov_quantile_predictor import create_predictor

# Создание предиктора на основе предустановки с измененными параметрами
predictor = create_predictor(
    model_type="enhanced_hybrid",
    preset_name="high_volatility",
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),  # Переопределяем квантили
    window_size=1000                           # Переопределяем размер окна
)
```

## Обучение и валидация моделей

### Процесс обучения и валидации

В контексте этого проекта "обучение" и "валидация" представляют собой по сути один и тот же процесс, но с разными наборами данных:

1. Модель обрабатывает данные последовательно (в режиме онлайн-обучения)
2. На каждом шаге:
   - Определяется текущее состояние рынка
   - Делается предсказание на основе предыдущих данных
   - Проверяется результат предсказания через N точек (prediction_depth)
   - Обновляется статистика и модели для следующих предсказаний

Таким образом, "валидация" - это просто запуск того же процесса на большем или отдельном наборе данных для оценки эффективности модели.

### Обучение и валидация модели

```python
# Стандартный запуск модели (обучение и проверка одновременно)
results = predictor.run_on_data(prices)

# Анализ результатов
success_rate = predictor.success_rate * 100
print(f"Успешность предсказаний: {success_rate:.2f}%")
```

### Использование раннего останова

Для ускорения процесса валидации на больших наборах данных можно использовать механизм раннего останова, который останавливает обработку данных, когда метрики перестают значительно меняться:

```python
from markov_quantile_predictor import PredictorConfig, EnhancedHybridPredictor
from markov_quantile_predictor.models.early_stopping import run_on_data_with_early_stopping
import types

# Создаем предиктор
config = PredictorConfig(...)
predictor = EnhancedHybridPredictor(config)

# Добавляем метод раннего останова
predictor.run_on_data_with_early_stopping = types.MethodType(run_on_data_with_early_stopping, predictor)

# Запускаем с ранним остановом
results = predictor.run_on_data_with_early_stopping(
    prices, 
    volumes=None,
    plateau_window_percent=5.0,  # Окно 5% от всех данных
    plateau_threshold=0.1,       # Изменение меньше 0.1% считается плато
    min_progress_percent=15.0    # Начинаем проверять плато после 15% данных
)
```

Параметры раннего останова:
- `plateau_window_percent` - размер окна в процентах для обнаружения плато
- `plateau_threshold` - пороговое значение изменения метрик для определения плато
- `min_progress_percent` - минимальный процент прогресса перед проверкой плато

## Примеры использования

### Базовый пример

```python
from markov_quantile_predictor import create_predictor
from markov_quantile_predictor.utils import load_data

# Загрузка данных
prices = load_data("data/example_data.csv")

# Создание предиктора
predictor = create_predictor(model_type="enhanced_hybrid", preset_name="standard")

# Запуск предсказания
results = predictor.run_on_data(prices)

# Визуализация результатов
predictor.visualize_results(prices, results, save_path="reports/results.png")

# Генерация отчета
predictor.generate_report(results, "reports/report.md", prices)
```

### Сравнение разных моделей

```python
from markov_quantile_predictor import create_predictor
from markov_quantile_predictor.utils import load_data

# Загрузка данных
prices = load_data("data/example_data.csv")

# Создание разных предикторов
markov_predictor = create_predictor(model_type="markov")
hybrid_predictor = create_predictor(model_type="hybrid")
enhanced_predictor = create_predictor(model_type="enhanced_hybrid")

# Запуск и сравнение результатов
markov_results = markov_predictor.run_on_data(prices)
hybrid_results = hybrid_predictor.run_on_data(prices)
enhanced_results = enhanced_predictor.run_on_data(prices)

print(f"MarkovPredictor успешность: {markov_predictor.success_rate*100:.2f}%")
print(f"MarkovQuantilePredictor успешность: {hybrid_predictor.success_rate*100:.2f}%")
print(f"EnhancedHybridPredictor успешность: {enhanced_predictor.success_rate*100:.2f}%")
```

### Валидация на большом наборе данных с ранним остановом

```python
from markov_quantile_predictor import PredictorConfig
from markov_quantile_predictor.models.hybrid_predictor import EnhancedHybridPredictor
from markov_quantile_predictor.models.early_stopping import run_on_data_with_early_stopping
from markov_quantile_predictor.utils import load_data
import types

# Загрузка данных
prices = load_data("data/large_dataset.csv")

# Создание предиктора
config = PredictorConfig(window_size=750, prediction_depth=15)
predictor = EnhancedHybridPredictor(config)

# Добавление функции раннего останова
predictor.run_on_data_with_early_stopping = types.MethodType(run_on_data_with_early_stopping, predictor)

# Запуск с ранним остановом
results = predictor.run_on_data_with_early_stopping(
    prices, 
    plateau_window_percent=5.0,
    plateau_threshold=0.1,
    min_progress_percent=15.0
)

# Анализ результатов
print(f"Успешность: {predictor.success_rate*100:.2f}%")
print(f"Обработано данных: {len(results)}/{len(prices)}")
```

## Анализ результатов

После выполнения предсказаний, можно анализировать результаты с помощью следующих методов:

```python
# Визуализация результатов
predictor.visualize_results(prices, results, save_path="reports/results.png")

# Генерация полного отчета
report = predictor.generate_report(results, "reports/report.md", prices)

# Получение статистики по состояниям
state_stats = predictor.get_state_statistics()
print(state_stats)

# Анализ квантильных предсказаний
quantile_results = [r for r in results if 'quantile_predictions' in r]
for r in quantile_results[:5]:
    print(f"Квантили: {r['quantile_predictions']}")
```

## Лицензия

MIT

## Автор

avrich5@pm.me