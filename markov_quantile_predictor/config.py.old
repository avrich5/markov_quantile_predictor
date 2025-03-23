"""
Модуль конфигурации для Markov Quantile Predictor.
"""

class PredictorConfig:
    """
    Конфигурация параметров предиктора.
    """
    def __init__(self, 
                 window_size=550,            # Размер окна для анализа (количество точек)
                 prediction_depth=15,        # Глубина предсказания (на сколько точек вперед)
                 min_confidence=0.80,        # Минимальная уверенность для принятия решения
                 state_length=4,             # Длина паттерна состояния
                 significant_change_pct=0.4,  # Порог значимого изменения в процентах
                 use_weighted_window=False,   # Использовать взвешенное окно
                 weight_decay=0.95,          # Коэффициент затухания весов
                 adaptive_weighting=False,   # Использовать адаптивное взвешивание
                 volatility_weighting=False, # Использовать взвешивание по волатильности
                 recency_boost=1.5,          # Множитель увеличения веса недавних событий
                 focus_on_best_states=False, # Фокус только на лучших состояниях
                 best_states=None,           # Список лучших состояний
                 quantiles=(0.1, 0.5, 0.9),  # Квантили для регрессии
                 min_samples_for_regression=10,  # Мин. кол-во наблюдений для регрессии
                 confidence_threshold=0.75,   # Порог уверенности для фильтрации предсказаний
                 max_coverage=0.05):          # Максимальное покрытие (доля предсказаний)
        
        self.window_size = window_size
        self.prediction_depth = prediction_depth
        self.min_confidence = min_confidence
        self.state_length = state_length
        self.significant_change_pct = significant_change_pct / 100  # Переводим проценты в доли
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