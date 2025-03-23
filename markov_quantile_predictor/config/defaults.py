"""
Дефолтные значения конфигурации для всего пакета markov_quantile_predictor.
Этот файл служит единой точкой определения всех параметров.
"""

# Параметры Марковского предиктора
DEFAULT_WINDOW_SIZE = 750
DEFAULT_PREDICTION_DEPTH = 15
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_STATE_LENGTH = 4
DEFAULT_SIGNIFICANT_CHANGE_PCT = 0.4  # в процентах
DEFAULT_USE_WEIGHTED_WINDOW = False
DEFAULT_WEIGHT_DECAY = 0.95
DEFAULT_ADAPTIVE_WEIGHTING = False
DEFAULT_VOLATILITY_WEIGHTING = False
DEFAULT_RECENCY_BOOST = 1.5
DEFAULT_FOCUS_ON_BEST_STATES = False
DEFAULT_BEST_STATES = []

# Параметры квантильной регрессии
DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_ALPHA = 0.1  # Регуляризация для квантильной регрессии
DEFAULT_MIN_SAMPLES_FOR_REGRESSION = 10

# Параметры гибридного предиктора
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_COVERAGE = 0.05  # 5% от общего количества точек

# Используемые модели
# Доступные опции: "markov", "quantile", "hybrid", "enhanced_hybrid"
DEFAULT_MODEL_TYPE = "enhanced_hybrid"

# Статистические параметры для отчетов
DEFAULT_SUCCESS_RATE_BASELINE = 50.0  # базовая линия успешности (случайное угадывание)
DEFAULT_CONFIDENCE_LEVELS = [0.01, 0.05, 0.1]  # уровни значимости для статистических тестов

# Параметры для работы с данными
DEFAULT_CSV_DELIMITER = ','
DEFAULT_CSV_ENCODING = 'utf-8'
DEFAULT_PRICE_COLUMN = 'price'
DEFAULT_DATE_COLUMN = 'date'
DEFAULT_VOLUME_COLUMN = 'volume'
DEFAULT_USE_VOLUMES = False

# Параметры визуализации
DEFAULT_FIGURE_SIZE = (15, 10)
DEFAULT_DPI = 100
DEFAULT_FONT_SIZE = 12
DEFAULT_COLOR_CORRECT = 'green'
DEFAULT_COLOR_INCORRECT = 'red'
DEFAULT_COLOR_UP = 'blue'
DEFAULT_COLOR_DOWN = 'orange'
DEFAULT_LINE_WIDTH = 2
DEFAULT_ALPHA = 0.7