"""
Markov Quantile Predictor - гибридная модель для предсказания временных рядов.
"""

# Сначала импортируем класс конфигурации из нового файла
from .predictor_config import PredictorConfig

# Импортируем предустановки из config
from .config import (
    STANDARD,
    HIGH_PRECISION,
    HIGH_COVERAGE,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    QUICK_TEST,
    create_config
)

# Затем импортируем модели
from .models.markov_predictor import MarkovPredictor
from .models.hybrid_predictor import MarkovQuantilePredictor, EnhancedHybridPredictor
from .models.quantile_regression import QuantileRegressionModel

# Импортируем утилиты
from .utils import load_data, ensure_dir, get_timestamp

# Импортируем фабрику
from .factory import create_predictor

__version__ = '0.2.0'