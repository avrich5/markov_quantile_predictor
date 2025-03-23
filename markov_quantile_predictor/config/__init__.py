"""
Модуль конфигурации для Markov Quantile Predictor.
Содержит константы по умолчанию и предустановленные конфигурации.
"""

from .defaults import *
from .presets import PRESETS, get_preset

# Экспортируем основные предустановки для удобного доступа
STANDARD = get_preset("standard")
HIGH_PRECISION = get_preset("high_precision")
HIGH_COVERAGE = get_preset("high_coverage")
HIGH_VOLATILITY = get_preset("high_volatility")
LOW_VOLATILITY = get_preset("low_volatility")
QUICK_TEST = get_preset("quick_test")

def create_config(preset_name=None, **kwargs):
    """
    Создает конфигурацию на основе предустановки с возможностью переопределения параметров
    
    Параметры:
    preset_name (str, optional): имя предустановки
    **kwargs: дополнительные параметры, которые переопределят параметры предустановки
    
    Возвращает:
    PredictorConfig: объект конфигурации
    """
    # Импортируем PredictorConfig здесь, чтобы избежать циклического импорта
    from ..predictor_config import PredictorConfig
    
    # Начинаем с параметров по умолчанию
    config_params = {
        "window_size": DEFAULT_WINDOW_SIZE,
        "prediction_depth": DEFAULT_PREDICTION_DEPTH,
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
        "state_length": DEFAULT_STATE_LENGTH,
        "significant_change_pct": DEFAULT_SIGNIFICANT_CHANGE_PCT,
        "use_weighted_window": DEFAULT_USE_WEIGHTED_WINDOW,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "adaptive_weighting": DEFAULT_ADAPTIVE_WEIGHTING,
        "volatility_weighting": DEFAULT_VOLATILITY_WEIGHTING,
        "recency_boost": DEFAULT_RECENCY_BOOST,
        "focus_on_best_states": DEFAULT_FOCUS_ON_BEST_STATES,
        "best_states": DEFAULT_BEST_STATES,
        "quantiles": DEFAULT_QUANTILES,
        "min_samples_for_regression": DEFAULT_MIN_SAMPLES_FOR_REGRESSION,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "max_coverage": DEFAULT_MAX_COVERAGE
    }
    
    # Если указана предустановка, используем её параметры
    if preset_name:
        preset = get_preset(preset_name)
        config_params.update(preset)
    
    # Переопределяем параметры из kwargs
    config_params.update(kwargs)
    
    # Создаем объект конфигурации
    return PredictorConfig(**config_params)