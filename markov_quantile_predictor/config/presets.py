"""
Предустановленные конфигурации для предикторов.
Эти предустановки включают оптимизированные параметры для различных сценариев.
"""

from .defaults import *

# Предустановка для стандартного использования
STANDARD_PRESET = {
    "window_size": DEFAULT_WINDOW_SIZE,  
    "prediction_depth": DEFAULT_PREDICTION_DEPTH,
    "min_confidence": DEFAULT_MIN_CONFIDENCE,
    "state_length": DEFAULT_STATE_LENGTH,
    "significant_change_pct": DEFAULT_SIGNIFICANT_CHANGE_PCT,
    "use_weighted_window": DEFAULT_USE_WEIGHTED_WINDOW,
    "quantiles": DEFAULT_QUANTILES,
    "min_samples_for_regression": DEFAULT_MIN_SAMPLES_FOR_REGRESSION,
    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "max_coverage": DEFAULT_MAX_COVERAGE
}

# Предустановка для высокой точности (меньше предсказаний, но более уверенных)
HIGH_PRECISION_PRESET = {
    "window_size": 1000,
    "prediction_depth": DEFAULT_PREDICTION_DEPTH,
    "min_confidence": 0.8,  # Повышенный порог уверенности
    "state_length": DEFAULT_STATE_LENGTH,
    "significant_change_pct": DEFAULT_SIGNIFICANT_CHANGE_PCT,
    "use_weighted_window": DEFAULT_USE_WEIGHTED_WINDOW,
    "quantiles": DEFAULT_QUANTILES,
    "min_samples_for_regression": 15,  # Больше образцов для обучения регрессии
    "confidence_threshold": 0.7,  # Повышенный порог для фильтрации предсказаний
    "max_coverage": 0.03  # Меньшее покрытие для более выборочных предсказаний
}

# Предустановка для высокого покрытия (больше предсказаний, но потенциально менее точных)
HIGH_COVERAGE_PRESET = {
    "window_size": 500,  # Меньшее окно для большего количества состояний
    "prediction_depth": DEFAULT_PREDICTION_DEPTH,
    "min_confidence": 0.4,  # Сниженный порог уверенности
    "state_length": 3,  # Меньшая длина состояния для большего покрытия
    "significant_change_pct": 0.3,  # Меньший порог для определения значимого изменения
    "use_weighted_window": DEFAULT_USE_WEIGHTED_WINDOW,
    "quantiles": DEFAULT_QUANTILES,
    "min_samples_for_regression": 5,  # Меньше образцов для создания регрессии
    "confidence_threshold": 0.3,  # Сниженный порог для фильтрации предсказаний
    "max_coverage": 0.1  # Повышенное покрытие
}

# Предустановка для работы с высокой волатильностью
HIGH_VOLATILITY_PRESET = {
    "window_size": DEFAULT_WINDOW_SIZE,
    "prediction_depth": DEFAULT_PREDICTION_DEPTH,
    "min_confidence": DEFAULT_MIN_CONFIDENCE,
    "state_length": DEFAULT_STATE_LENGTH,
    "significant_change_pct": 0.8,  # Повышенный порог для определения значимого изменения
    "use_weighted_window": True,  # Включено взвешивание
    "weight_decay": 0.9,  # Быстрое затухание весов для фокуса на недавних событиях
    "adaptive_weighting": True,  # Адаптивное взвешивание
    "volatility_weighting": True,  # Взвешивание по волатильности
    "recency_boost": 2.0,  # Повышенный бустинг недавних событий
    "quantiles": (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),  # Расширенный набор квантилей
    "min_samples_for_regression": DEFAULT_MIN_SAMPLES_FOR_REGRESSION,
    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "max_coverage": DEFAULT_MAX_COVERAGE
}

# Предустановка для низкой волатильности / бокового рынка
LOW_VOLATILITY_PRESET = {
    "window_size": 1000,  # Расширенное окно для захвата большей истории
    "prediction_depth": DEFAULT_PREDICTION_DEPTH,
    "min_confidence": DEFAULT_MIN_CONFIDENCE,
    "state_length": 5,  # Увеличенная длина состояния для более точного определения паттернов
    "significant_change_pct": 0.2,  # Сниженный порог для определения значимого изменения
    "use_weighted_window": DEFAULT_USE_WEIGHTED_WINDOW,
    "quantiles": (0.05, 0.25, 0.5, 0.75, 0.95),  # Упрощенный набор квантилей
    "min_samples_for_regression": DEFAULT_MIN_SAMPLES_FOR_REGRESSION,
    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "max_coverage": DEFAULT_MAX_COVERAGE
}

# Предустановка для быстрого прототипирования / тестирования
QUICK_TEST_PRESET = {
    "window_size": 250,  # Уменьшенное окно для быстрого выполнения
    "prediction_depth": 5,  # Короткая глубина предсказания
    "min_confidence": 0.5,
    "state_length": 3,
    "significant_change_pct": DEFAULT_SIGNIFICANT_CHANGE_PCT,
    "use_weighted_window": False,
    "quantiles": (0.1, 0.5, 0.9),  # Минимальный набор квантилей
    "min_samples_for_regression": 5,
    "confidence_threshold": 0.5,
    "max_coverage": 0.1
}

# Словарь всех предустановок для удобного доступа
PRESETS = {
    "standard": STANDARD_PRESET,
    "high_precision": HIGH_PRECISION_PRESET,
    "high_coverage": HIGH_COVERAGE_PRESET,
    "high_volatility": HIGH_VOLATILITY_PRESET,
    "low_volatility": LOW_VOLATILITY_PRESET,
    "quick_test": QUICK_TEST_PRESET
}

def get_preset(name):
    """
    Получает предустановку по имени
    
    Параметры:
    name (str): имя предустановки
    
    Возвращает:
    dict: словарь с параметрами предустановки
    """
    if name in PRESETS:
        return PRESETS[name].copy()
    else:
        raise ValueError(f"Неизвестная предустановка: {name}. "
                         f"Доступные предустановки: {', '.join(PRESETS.keys())}")