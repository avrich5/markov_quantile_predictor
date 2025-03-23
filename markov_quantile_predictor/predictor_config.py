"""
Модуль конфигурации для Markov Quantile Predictor.
"""

from markov_quantile_predictor.config.defaults import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_PREDICTION_DEPTH,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_STATE_LENGTH,
    DEFAULT_SIGNIFICANT_CHANGE_PCT,
    DEFAULT_USE_WEIGHTED_WINDOW,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_ADAPTIVE_WEIGHTING,
    DEFAULT_VOLATILITY_WEIGHTING,
    DEFAULT_RECENCY_BOOST,
    DEFAULT_FOCUS_ON_BEST_STATES,
    DEFAULT_BEST_STATES,
    DEFAULT_QUANTILES,
    DEFAULT_MIN_SAMPLES_FOR_REGRESSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_COVERAGE
)

class PredictorConfig:
    """
    Конфигурация параметров предиктора.
    """
    def __init__(self, 
                 window_size=DEFAULT_WINDOW_SIZE,
                 prediction_depth=DEFAULT_PREDICTION_DEPTH,
                 min_confidence=DEFAULT_MIN_CONFIDENCE,
                 state_length=DEFAULT_STATE_LENGTH,
                 significant_change_pct=DEFAULT_SIGNIFICANT_CHANGE_PCT,
                 use_weighted_window=DEFAULT_USE_WEIGHTED_WINDOW,
                 weight_decay=DEFAULT_WEIGHT_DECAY,
                 adaptive_weighting=DEFAULT_ADAPTIVE_WEIGHTING,
                 volatility_weighting=DEFAULT_VOLATILITY_WEIGHTING,
                 recency_boost=DEFAULT_RECENCY_BOOST,
                 focus_on_best_states=DEFAULT_FOCUS_ON_BEST_STATES,
                 best_states=None,
                 quantiles=DEFAULT_QUANTILES,
                 min_samples_for_regression=DEFAULT_MIN_SAMPLES_FOR_REGRESSION,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                 max_coverage=DEFAULT_MAX_COVERAGE):
        
        self.window_size = window_size
        self.prediction_depth = prediction_depth
        self.min_confidence = min_confidence
        self.state_length = state_length
        self.significant_change_pct = significant_change_pct / 100 if significant_change_pct > 1 else significant_change_pct  # Переводим проценты в доли если нужно
        self.use_weighted_window = use_weighted_window
        self.weight_decay = weight_decay
        self.adaptive_weighting = adaptive_weighting
        self.volatility_weighting = volatility_weighting
        self.recency_boost = recency_boost
        self.focus_on_best_states = focus_on_best_states
        self.best_states = best_states if best_states else []
        self.quantiles = quantiles
        self.min_samples_for_regression = min_samples_for_regression
        self.confidence_threshold = confidence_threshold
        self.max_coverage = max_coverage
        
    def __str__(self):
        """Строковое представление конфигурации"""
        config_str = (f"PredictorConfig(window_size={self.window_size}, "
                     f"prediction_depth={self.prediction_depth}, "
                     f"min_confidence={self.min_confidence}, "
                     f"state_length={self.state_length}, "
                     f"significant_change_pct={self.significant_change_pct*100}%, "
                     f"use_weighted_window={self.use_weighted_window}, "
                     f"weight_decay={self.weight_decay}")
        
        if self.adaptive_weighting:
            config_str += f", adaptive_weighting=True"
        
        if self.volatility_weighting:
            config_str += f", volatility_weighting=True"
            
        if self.recency_boost != 1.0:
            config_str += f", recency_boost={self.recency_boost}"
        
        if self.focus_on_best_states:
            config_str += f", focus_on_best_states=True, best_states={self.best_states}"
            
        config_str += f", quantiles={self.quantiles}"
        config_str += f", min_samples_for_regression={self.min_samples_for_regression}"
        config_str += f", confidence_threshold={self.confidence_threshold}"
        config_str += f", max_coverage={self.max_coverage}"
        
        config_str += ")"
        return config_str
    
    def to_dict(self):
        """
        Преобразует конфигурацию в словарь
        
        Возвращает:
        dict: словарь с параметрами конфигурации
        """
        return {
            "window_size": self.window_size,
            "prediction_depth": self.prediction_depth,
            "min_confidence": self.min_confidence,
            "state_length": self.state_length,
            "significant_change_pct": self.significant_change_pct * 100,  # Переводим обратно в проценты
            "use_weighted_window": self.use_weighted_window,
            "weight_decay": self.weight_decay,
            "adaptive_weighting": self.adaptive_weighting,
            "volatility_weighting": self.volatility_weighting,
            "recency_boost": self.recency_boost,
            "focus_on_best_states": self.focus_on_best_states,
            "best_states": self.best_states,
            "quantiles": self.quantiles,
            "min_samples_for_regression": self.min_samples_for_regression,
            "confidence_threshold": self.confidence_threshold,
            "max_coverage": self.max_coverage
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Создает конфигурацию из словаря
        
        Параметры:
        config_dict (dict): словарь с параметрами конфигурации
        
        Возвращает:
        PredictorConfig: объект конфигурации
        """
        return cls(**config_dict)