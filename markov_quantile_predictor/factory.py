"""
Фабрика для создания различных предикторов.
Предоставляет единый интерфейс для создания моделей разных типов.
"""

from .predictor_config import PredictorConfig
from .config import defaults, create_config
from .models import MarkovPredictor, MarkovQuantilePredictor, EnhancedHybridPredictor

def create_predictor(model_type=None, config=None, preset_name=None, **kwargs):
    """
    Создает предиктор указанного типа с заданной конфигурацией
    
    Параметры:
    model_type (str, optional): тип модели ("markov", "quantile", "hybrid", "enhanced_hybrid")
    config (PredictorConfig, optional): объект конфигурации
    preset_name (str, optional): имя предустановки для конфигурации
    **kwargs: дополнительные параметры для конфигурации
    
    Возвращает:
    Предиктор соответствующего типа
    """
    # Если тип модели не указан, используем значение по умолчанию
    if model_type is None:
        model_type = defaults.DEFAULT_MODEL_TYPE
    
    # Проверяем, что тип модели допустимый
    valid_types = ["markov", "quantile", "hybrid", "enhanced_hybrid"]
    if model_type not in valid_types:
        raise ValueError(f"Неизвестный тип модели: {model_type}. "
                        f"Допустимые типы: {', '.join(valid_types)}")
    
    # Если конфигурация не предоставлена, создаем ее
    if config is None:
        config = create_config(preset_name, **kwargs)
    
    # Создаем предиктор нужного типа
    if model_type == "markov":
        return MarkovPredictor(config)
    elif model_type == "quantile" or model_type == "hybrid":
        return MarkovQuantilePredictor(config)
    elif model_type == "enhanced_hybrid":
        return EnhancedHybridPredictor(config)