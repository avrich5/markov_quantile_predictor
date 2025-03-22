"""
Markov Quantile Predictor - гибридный предиктор временных рядов,
объединяющий марковские цепи и квантильную регрессию.
"""

from .config import PredictorConfig
from .markov_predictor import MarkovPredictor
# from .quantile_regression import QuantileRegressionModel
# from .hybrid_predictor import MarkovQuantilePredictor
from .quantile_regression import QuantileRegressionModel, EnhancedQuantileRegressionModel
from .hybrid_predictor import MarkovQuantilePredictor, EnhancedHybridPredictor

__version__ = '0.1.0'