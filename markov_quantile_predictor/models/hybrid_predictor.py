"""
Модуль с реализацией гибридного предиктора на основе марковского процесса и квантильной регрессии.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from .markov_predictor import MarkovPredictor
from .quantile_regression import QuantileRegressionModel


class MarkovQuantilePredictor(MarkovPredictor):
    """
    Гибридный предиктор, объединяющий марковский предиктор и квантильную регрессию
    в последовательной архитектуре.
    """
    
    def __init__(self, config=None):
        """
        Инициализация предиктора
        
        Параметры:
        config (PredictorConfig): конфигурация параметров предиктора
        """
        super().__init__(config)
        
        # Словарь для хранения моделей квантильной регрессии для разных состояний
        self.quantile_models = {}
        
        # Базовая квантильная регрессия без привязки к состояниям
        self.base_quantile_model = QuantileRegressionModel(
            quantiles=self.config.quantiles,
            alpha=0.1
        )
    
    def _extract_features(self, prices, idx):
        """
        Извлекает признаки для квантильной регрессии
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): индекс точки
        
        Возвращает:
        numpy.array: вектор признаков
        """
        # Используем только данные внутри текущего окна
        start_idx = max(0, idx - self.config.window_size)
        window = prices[start_idx:idx+1]
        
        if len(window) < 3:
            return np.array([0, 0, 0])
        
        # Вычисляем признаки
        
        # 1. Последнее изменение
        recent_change = prices[idx] / prices[idx-1] - 1 if prices[idx-1] != 0 else 0
        
        # 2. Волатильность (стандартное отклонение изменений в окне)
        changes = np.array([
            prices[i+1]/prices[i] - 1 
            for i in range(start_idx, idx) 
            if prices[i] != 0 and i+1 <= idx
        ])
        volatility = np.std(changes) if len(changes) > 0 else 0
        
        # 3. Отклонение от скользящей средней
        ma_period = min(20, len(window))
        if ma_period > 0:
            sma = np.mean(window[-ma_period:])
            deviation = prices[idx] / sma - 1 if sma != 0 else 0
        else:
            deviation = 0
        
        # 4. Средняя скорость изменения (угол наклона)
        if len(window) >= 5:
            speed = (prices[idx] / prices[idx-5] - 1) / 5 if prices[idx-5] != 0 else 0
        else:
            speed = 0
        
        # 5. Ускорение изменения (разница между текущей и предыдущей скоростью)
        if len(window) >= 10:
            prev_speed = (prices[idx-5] / prices[idx-10] - 1) / 5 if prices[idx-10] != 0 else 0
            acceleration = speed - prev_speed
        else:
            acceleration = 0
        
        return np.array([recent_change, volatility, deviation, speed, acceleration])
    
    def _collect_state_samples(self, prices, idx, dynamic_threshold):
        """
        Собирает исторические данные для текущего состояния
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        dynamic_threshold (float): динамический порог
        
        Возвращает:
        tuple: (X, y, current_features)
        """
        # Получаем текущее состояние
        current_state = self._get_state(prices, idx, dynamic_threshold)
        if current_state is None:
            return None, None, None
        
        # Определяем окно для анализа
        start_idx = max(self.config.state_length, idx - self.config.window_size)
        
        # Собираем данные для этого состояния
        X = []
        y = []
        
        for i in range(start_idx, idx - self.config.prediction_depth + 1):
            # Определяем состояние в этой точке
            state = self._get_state(prices, i, dynamic_threshold)
            if state != current_state:
                continue
            
            # Извлекаем признаки
            features = self._extract_features(prices, i)
            
            # Определяем целевое значение (процентное изменение через prediction_depth точек)
            if i + self.config.prediction_depth < len(prices):
                future_price = prices[i + self.config.prediction_depth]
                current_price = prices[i]
                if current_price != 0:
                    pct_change = future_price / current_price - 1
                    
                    X.append(features)
                    y.append(pct_change)
        
        # Извлекаем признаки для текущей точки
        current_features = self._extract_features(prices, idx)
        
        return np.array(X), np.array(y), current_features
        
def visualize_results(self, prices, results, save_path=None):
    """
    Визуализирует результаты предсказаний, включая информацию от квантильной регрессии
    с безопасной обработкой различных наборов квантилей
    
    Параметры:
    prices (numpy.array): массив цен
    results (list): результаты предсказаний
    save_path (str): путь для сохранения графиков
    """
    # Сначала используем метод визуализации родительского класса
    super().visualize_results(prices, results, save_path)
    
    # Дополнительно создаем график с информацией от квантильной регрессии
    quantile_results = [r for r in results if 'quantile_predictions' in r and r['quantile_predictions']]
    
    if not quantile_results:
        return  # Нет данных для визуализации
    
    # Определим доступные квантили из первого результата с непустыми предсказаниями
    first_result = next((r for r in quantile_results if r['quantile_predictions']), None)
    if not first_result:
        return
        
    available_quantiles = sorted(first_result['quantile_predictions'].keys())
    if not available_quantiles:
        return
    
    # Найдем ближайшие доступные квантили к нужным значениям
    median_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.5))
    lower_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.1))
    upper_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.9))
    
    # Создаем новый график
    plt.figure(figsize=(15, 8))
    
    # График цен
    plt.plot(prices, color='blue', alpha=0.5, label='Цена')
    
    # Собираем данные для отображения квантилей
    indices = []
    medians = []
    lower_bounds_90 = []
    upper_bounds_90 = []
    
    # Если есть крайние квантили (близкие к 0.05 и 0.95), используем их тоже
    has_extreme_quantiles = len(available_quantiles) >= 4  # Минимум 4 квантиля для крайних значений
    
    # Находим крайние квантили, если они есть
    extreme_lower_q = None
    extreme_upper_q = None
    
    if has_extreme_quantiles:
        # Используем самый нижний и самый верхний доступные квантили
        extreme_lower_q = min(available_quantiles)
        extreme_upper_q = max(available_quantiles)
        
        # Если они совпадают с квантилями 10/90, то не используем их отдельно
        if extreme_lower_q == lower_q:
            extreme_lower_q = None
        if extreme_upper_q == upper_q:
            extreme_upper_q = None
            
        # Если оба крайних квантиля совпадают с основными, отключаем отрисовку крайних
        if extreme_lower_q is None and extreme_upper_q is None:
            has_extreme_quantiles = False
    
    lower_bounds_extreme = []
    upper_bounds_extreme = []

    for r in quantile_results:
        q_preds = r['quantile_predictions']
        if not q_preds:
            continue
            
        idx = r['index']
        indices.append(idx)
        
        # Преобразуем из процентного изменения в абсолютное значение цены
        price = prices[idx]
        medians.append(price * (1 + q_preds[median_q]))
        lower_bounds_90.append(price * (1 + q_preds[lower_q]))
        upper_bounds_90.append(price * (1 + q_preds[upper_q]))
        
        # Добавляем крайние квантили, если они доступны
        if has_extreme_quantiles:
            if extreme_lower_q is not None:
                lower_bounds_extreme.append(price * (1 + q_preds[extreme_lower_q]))
            if extreme_upper_q is not None:
                upper_bounds_extreme.append(price * (1 + q_preds[extreme_upper_q]))

    # Отображаем медианные предсказания
    plt.scatter(indices, medians, color='green', s=30, alpha=0.7, label='Медианное предсказание')

    # Отображаем интервалы предсказаний [10%, 90%]
    for i in range(len(indices)):
        plt.plot([indices[i], indices[i]], [lower_bounds_90[i], upper_bounds_90[i]], 
                color='green', alpha=0.3)

    # Отображаем интервалы предсказаний для крайних квантилей, если доступны
    if has_extreme_quantiles and extreme_lower_q is not None and extreme_upper_q is not None:
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]], [lower_bounds_extreme[i], upper_bounds_extreme[i]], 
                    color='blue', alpha=0.2, linestyle='--')
        
    plt.title('Предсказания с квантильными интервалами')
    plt.xlabel('Индекс')
    plt.ylabel('Цена')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Сохраняем график, если указан путь
    if save_path:
        quantile_path = save_path.replace('.png', '_quantiles.png')
        plt.savefig(quantile_path)
    
    plt.show()

def generate_report(self, results, save_path=None, prices=None):
    """
    Генерирует отчет о результатах предсказаний, включая информацию от квантильной регрессии
    с безопасной обработкой различных наборов квантилей
    
    Параметры:
    results (list): результаты предсказаний
    save_path (str): путь для сохранения отчета
    prices (numpy.array, optional): массив цен, используемый для анализа фактических изменений
    
    Возвращает:
    str: текст отчета
    """
    # Получаем базовый отчет от родительского класса
    # base_report = super().generate_report(results, None)
    base_report = super().generate_report(results, None, prices)

    
    # Собираем статистику по квантильным предсказаниям
    quantile_results = [r for r in results if 'quantile_predictions' in r and r['quantile_predictions']]
    
    if not quantile_results or prices is None:
        # Если нет данных от квантильной регрессии или не передан массив цен
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(base_report)
        return base_report
    
    # Определим доступные квантили из первого результата с непустыми предсказаниями
    first_result = next((r for r in quantile_results if r['quantile_predictions']), None)
    if not first_result:
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(base_report)
        return base_report
        
    available_quantiles = sorted(first_result['quantile_predictions'].keys())
    if not available_quantiles:
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(base_report)
        return base_report
    
    # Находим ближайшие доступные квантили к стандартным значениям
    lower_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.1))
    upper_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.9))
    mid_lower_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.25))
    mid_upper_q = min(available_quantiles, key=lambda q: abs(float(q) - 0.75))
    
    # Если доступно больше 4 квантилей, используем крайние для [5%, 95%]
    extreme_lower_q = None
    extreme_upper_q = None
    
    if len(available_quantiles) >= 4:
        # Для крайних значений используем минимальный и максимальный доступные квантили
        extreme_lower_q = min(available_quantiles)
        extreme_upper_q = max(available_quantiles)
        
        # Если они совпадают с основными, то не используем их отдельно
        if extreme_lower_q == lower_q:
            extreme_lower_q = None
        if extreme_upper_q == upper_q:
            extreme_upper_q = None
    
    # Анализируем результаты квантильных предсказаний
    
    # 1. Точность квантильных предсказаний
    # Считаем сколько раз фактическое значение попало в разные интервалы
    actual_in_interval_90 = 0  # [10%, 90%]
    actual_in_interval_mid = 0  # [25%, 75%]
    actual_in_interval_extreme = 0  # [5%, 95%] или крайние доступные

    for r in quantile_results:
        idx = r['index']
        if idx + self.config.prediction_depth >= len(prices):
            continue
                
        # Фактическое изменение
        actual_change = prices[idx + self.config.prediction_depth] / prices[idx] - 1
        
        # Предсказанные квантили
        q_preds = r['quantile_predictions']
        
        # Проверяем, попало ли фактическое изменение в разные интервалы
        if q_preds[lower_q] <= actual_change <= q_preds[upper_q]:
            actual_in_interval_90 += 1
        if q_preds[mid_lower_q] <= actual_change <= q_preds[mid_upper_q]:
            actual_in_interval_mid += 1
        # Если есть крайние квантили
        if extreme_lower_q is not None and extreme_upper_q is not None:
            if q_preds[extreme_lower_q] <= actual_change <= q_preds[extreme_upper_q]:
                actual_in_interval_extreme += 1

    # Вычисляем процент попаданий в разные интервалы
    valid_results_count = len([r for r in quantile_results if 
                           r['index'] + self.config.prediction_depth < len(prices)])
    
    interval_coverage_90 = (actual_in_interval_90 / valid_results_count * 100 
                         if valid_results_count > 0 else 0)
    interval_coverage_mid = (actual_in_interval_mid / valid_results_count * 100 
                          if valid_results_count > 0 else 0)
    
    if extreme_lower_q is not None and extreme_upper_q is not None:
        interval_coverage_extreme = (actual_in_interval_extreme / valid_results_count * 100 
                                  if valid_results_count > 0 else 0)
    else:
        interval_coverage_extreme = None

    # 2. Средние предсказанные изменения для разных квантилей
    mean_quantiles = {}
    for q in available_quantiles:
        mean_quantiles[q] = np.mean([r['quantile_predictions'][q] * 100 for r in quantile_results])
    
    # 3. Ширина предсказанных интервалов
    mean_interval_width_90 = np.mean([(r['quantile_predictions'][upper_q] - 
                                    r['quantile_predictions'][lower_q]) * 100 
                                   for r in quantile_results])
    mean_interval_width_50 = np.mean([(r['quantile_predictions'][mid_upper_q] - 
                                    r['quantile_predictions'][mid_lower_q]) * 100 
                                   for r in quantile_results])
    
    if extreme_lower_q is not None and extreme_upper_q is not None:
        mean_interval_width_extreme = np.mean([(r['quantile_predictions'][extreme_upper_q] - 
                                            r['quantile_predictions'][extreme_lower_q]) * 100 
                                           for r in quantile_results])
    else:
        mean_interval_width_extreme = None

    # Формируем дополнительный отчет
    quantile_report = f"""
## Статистика квантильной регрессии

### Общая информация
- Количество предсказаний с квантильной регрессией: {len(quantile_results)}
- Доступные квантили: {', '.join([f"{q:.2f}" for q in available_quantiles])}
- Процент попаданий фактического значения в интервал [{lower_q:.2f}, {upper_q:.2f}]: {interval_coverage_90:.2f}%
- Процент попаданий фактического значения в интервал [{mid_lower_q:.2f}, {mid_upper_q:.2f}]: {interval_coverage_mid:.2f}%
"""

    # Добавляем информацию о крайних квантилях, если они доступны
    if extreme_lower_q is not None and extreme_upper_q is not None:
        quantile_report += f"- Процент попаданий фактического значения в интервал [{extreme_lower_q:.2f}, {extreme_upper_q:.2f}]: {interval_coverage_extreme:.2f}%\n"

    quantile_report += "\n### Средние предсказанные изменения\n"
    
    # Добавляем средние значения по всем доступным квантилям
    for q in sorted(available_quantiles):
        quantile_report += f"- Средний квантиль {q:.2f}: {mean_quantiles[q]:.2f}%\n"

    quantile_report += "\n### Ширина предсказанных интервалов\n"
    quantile_report += f"- Средняя ширина интервала [{lower_q:.2f}, {upper_q:.2f}]: {mean_interval_width_90:.2f}%\n"
    quantile_report += f"- Средняя ширина интервала [{mid_lower_q:.2f}, {mid_upper_q:.2f}]: {mean_interval_width_50:.2f}%\n"
    
    # Добавляем информацию о ширине крайнего интервала, если доступен
    if extreme_lower_q is not None and extreme_upper_q is not None and mean_interval_width_extreme is not None:
        quantile_report += f"- Средняя ширина интервала [{extreme_lower_q:.2f}, {extreme_upper_q:.2f}]: {mean_interval_width_extreme:.2f}%\n"

    quantile_report += f"""
### Квантильные модели
- Количество обученных моделей (по состояниям): {len(self.quantile_models)}
"""
    
    # Объединяем отчеты
    full_report = base_report + quantile_report
    
    # Сохраняем отчет, если указан путь
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
    
    return full_report


############### --------- EnhancedHybridPredictor(MarkovQuantilePredictor) ------- ###############

class EnhancedHybridPredictor(MarkovQuantilePredictor):
    """
    Улучшенный гибридный предиктор, использующий характеристики состояний
    как входные данные для квантильной регрессии (Вариант 3).
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        print(f"DEBUG: Config params - confidence_threshold: {self.config.confidence_threshold}, quantiles: {self.config.quantiles}")
        self.feature_extractors = self._init_feature_extractors()

        # Проверьте, что базовая модель правильно инициализирована
        self.base_quantile_model = QuantileRegressionModel(
            quantiles=self.config.quantiles,
            alpha=0.1
        )
        print(f"Initialized with quantiles: {self.config.quantiles}")

    def _init_feature_extractors(self):
        """Инициализирует извлекатели признаков для квантильной регрессии"""
        extractors = {
            'basic': self._extract_basic_features,
            'market': self._extract_market_features,
            'state': self._extract_state_features
        }
        return extractors
    
    def _extract_basic_features(self, prices, idx):
        """Базовые признаки цены (уже реализовано в текущей версии)"""
        # Используем существующую реализацию _extract_features
        return super()._extract_features(prices, idx)
    
    def _extract_market_features(self, prices, volumes, idx):
        """Дополнительные рыночные признаки (объемы, индикаторы)"""
        features = []
        
        # Получаем данные из окна
        start_idx = max(0, idx - self.config.window_size)
        price_window = prices[start_idx:idx+1]
        
        # 1. Относительный объем (если доступен)
        if volumes is not None and len(volumes) > idx:
            vol_window = volumes[start_idx:idx+1]
            if len(vol_window) > 0 and np.mean(vol_window) > 0:
                rel_volume = volumes[idx] / np.mean(vol_window[-20:]) if len(vol_window) >= 20 else 1.0
                features.append(rel_volume)
            else:
                features.append(1.0)
        else:
            features.append(1.0)
        
        # 2. RSI (если окно достаточно большое)
        if len(price_window) >= 14:
            try:
                changes = np.diff(price_window)
                gains = np.array([max(0, change) for change in changes])
                losses = np.array([max(0, -change) for change in changes])
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                
                features.append(rsi / 100)  # нормализуем к [0,1]
            except:
                features.append(0.5)  # среднее значение при ошибке
        else:
            features.append(0.5)
        
        return np.array(features)
    
    def _extract_state_features(self, prices, idx, dynamic_threshold):
        """Извлекает признаки, связанные с текущим марковским состоянием"""
        current_state = self._get_state(prices, idx, dynamic_threshold)
        if current_state is None:
            # Возвращаем нулевой вектор если состояние не определено
            return np.zeros(3 + self.config.state_length)
        
        features = []
        
        # 1. Статистика по текущему состоянию (из исторических данных)
        start_idx = max(self.config.state_length, idx - self.config.window_size)
        transitions = defaultdict(lambda: {1: 0, 2: 0})
        
        # Собираем статистику переходов
        for i in range(start_idx, idx - self.config.prediction_depth + 1):
            state = self._get_state(prices, i, dynamic_threshold)
            if state != current_state:
                continue
            
            outcome = self._determine_outcome(prices, i, dynamic_threshold)
            if outcome is None or outcome == 0:
                continue
            
            transitions[state][outcome] += 1
        
        # Рассчитываем вероятности переходов
        total = sum(transitions[current_state].values()) if current_state in transitions else 0
        up_prob = transitions[current_state].get(1, 0) / total if total > 0 else 0.5
        down_prob = transitions[current_state].get(2, 0) / total if total > 0 else 0.5
        
        features.append(up_prob)
        features.append(down_prob)
        features.append(total / self.config.window_size)  # нормализованная частота состояния
        
        # 2. Кодирование самого состояния (one-hot или непосредственно)
        for movement in current_state:
            features.append(float(movement) / 2.0)  # нормализуем 1->0.5, 2->1.0
        
        return np.array(features)
    
    def _collect_enhanced_features(self, prices, volumes, idx, dynamic_threshold):
        """Собирает все признаки для гибридной модели"""
        basic_features = self._extract_basic_features(prices, idx)
        market_features = self._extract_market_features(prices, volumes, idx)
        state_features = self._extract_state_features(prices, idx, dynamic_threshold)
        
        # Объединяем все признаки в один вектор
        combined_features = np.concatenate([basic_features, market_features, state_features])
        return combined_features
    

    def predict_at_point(self, prices, volumes=None, idx=None, max_predictions=None, current_predictions=0):
        # Базовые проверки (как в родительском классе)
        if idx % 1000 == 0:
            print(f"DEBUG: predict_at_point at idx={idx}, max_predictions={max_predictions}, current_predictions={current_predictions}")
        
        if idx < self.config.window_size or idx < self.config.state_length:
            return {'prediction': 0, 'confidence': 0.0}
        
        if max_predictions is not None and current_predictions >= max_predictions:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Вычисляем динамический порог
        dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
        
        # Получаем текущее состояние
        current_state = self._get_state(prices, idx, dynamic_threshold)
        if current_state is None:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Отладочный вывод 1 раз на 1000 точек
        if idx % 1000 == 0:
            print(f"Debug at idx={idx}: State={current_state}, Threshold={dynamic_threshold:.6f}")
        
        # Собираем все признаки для гибридной модели
        features = self._collect_enhanced_features(prices, volumes, idx, dynamic_threshold)
        
        # Если у нас есть обученная модель для текущего состояния, используем ее
        if current_state in self.quantile_models:
            model = self.quantile_models[current_state]
            predictions = model.predict_single(features)
            if idx % 1000 == 0:
                print(f"  Using state model, predictions: {predictions}")
        else:
            # Если модели нет, используем базовую модель (или None)
            if self.base_quantile_model.is_fitted:
                predictions = self.base_quantile_model.predict_single(features)
                if idx % 1000 == 0:
                    print(f"  Using base model, predictions: {predictions}")
            else:
                if idx % 1000 == 0:
                    print(f"  No models available")
                return {'prediction': 0, 'confidence': 0.0}
        
        # Если не удалось получить предсказания, возвращаем пустой результат
        if predictions is None:
            if idx % 1000 == 0:
                print(f"  Predictions is None")
            return {'prediction': 0, 'confidence': 0.0}
        
        # Определяем квантили, которые нам нужны
        available_quantiles = sorted(predictions.keys())
        
        # Находим медиану и нужные квантили для расчета уверенности
        median_q = min(available_quantiles, key=lambda q: abs(q - 0.5))
        median = predictions[median_q]
        
        lower_q = min(available_quantiles, key=lambda q: abs(q - 0.1))
        lower = predictions[lower_q]
        
        upper_q = min(available_quantiles, key=lambda q: abs(q - 0.9))
        upper = predictions[upper_q]
        
        # Отладочная информация
        if idx % 1000 == 0:
            print(f"  Median: {median:.6f}, Lower: {lower:.6f}, Upper: {upper:.6f}")
        
        # Определяем направление
        prediction = 0
        if median > dynamic_threshold:
            prediction = 1  # рост
        elif median < -dynamic_threshold:
            prediction = 2  # падение
        
        # Рассчитываем уверенность на основе распределения квантилей (оригинальный метод)
        if prediction == 1:  # Рост
            confidence = min(1.0, max(0, lower / dynamic_threshold + 0.5))
        elif prediction == 2:  # Падение
            confidence = min(1.0, max(0, -upper / dynamic_threshold + 0.5))
        else:
            confidence = 0.0
        
        # Отладочная информация о решении
        if idx % 1000 == 0:
            print(f"  Decision: prediction={prediction}, confidence={confidence:.4f}, threshold={self.config.confidence_threshold}")
        
        # Отладочная информация о решении
        if idx % 1000 == 0:
            print(f"  Decision: prediction={prediction}, confidence={confidence:.4f}, threshold={self.config.confidence_threshold}")
            # Добавьте эту строку для отладки
            print(f"  Debug config values: confidence_threshold={self.config.confidence_threshold}, max_coverage={self.config.max_coverage}")

        # Применяем фильтр уверенности
        if confidence < self.config.confidence_threshold:
            prediction = 0
            confidence = 0.0
        
        # Формируем итоговый результат
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'state': current_state,
            'quantile_predictions': {q: float(val) for q, val in predictions.items()} if predictions else {},
            'features': features.tolist() if isinstance(features, np.ndarray) else []  # для отладки
        }
        
        return result

        
    def run_on_data(self, prices, volumes=None, verbose=True):
        """
        Запускает гибридную модель на всем наборе данных
        
        Параметры:
        prices (numpy.array): массив цен
        volumes (numpy.array, optional): массив объемов торгов
        verbose (bool): выводить информацию о прогрессе
        
        Возвращает:
        list: результаты предсказаний
        """
        # Предварительно вычисляем изменения цен
        self._precompute_changes(prices)
        
        results = []
        
        # Начинаем с точки, где у нас достаточно данных для анализа
        min_idx = max(self.config.window_size, self.config.state_length)
        
        # Вычисляем максимальное количество предсказаний
        max_predictions = int(len(prices) * self.config.max_coverage)
        current_predictions = 0
        
        # Проходим по всем точкам
        with tqdm(total=len(prices) - min_idx - self.config.prediction_depth, 
                    desc="Processing", disable=not verbose) as pbar:
            for idx in range(min_idx, len(prices) - self.config.prediction_depth):
                # Делаем предсказание
                pred_result = self.predict_at_point(prices, volumes, idx, max_predictions, current_predictions)
                prediction = pred_result['prediction']
                
                # Получаем состояние для этой точки и сохраняем его, даже если предсказание не делается
                dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
                current_state = self._get_state(prices, idx, dynamic_threshold)
                
                # Если предсказание не "не знаю", проверяем результат
                if prediction != 0:
                    current_predictions += 1
                    actual_outcome = self._determine_outcome(prices, idx, dynamic_threshold)
                    
                    # Пропускаем проверку, если результат незначительное изменение (0)
                    if actual_outcome is None or actual_outcome == 0:
                        pbar.update(1)
                        continue
                    
                    is_correct = (prediction == actual_outcome)
                    
                    # Обновляем статистику
                    self.total_predictions += 1
                    if is_correct:
                        self.correct_predictions += 1
                    
                    # Обновляем успешность
                    self.success_rate = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
                    
                    # Сохраняем статистику для этой точки
                    self.point_statistics[idx] = {
                        'correct': self.correct_predictions,
                        'total': self.total_predictions,
                        'success_rate': self.success_rate
                    }
                    
                    # Обновляем статистику по этому состоянию
                    if current_state:  # Проверяем, что состояние определено
                        self.state_statistics[current_state]['total'] += 1
                        if is_correct:
                            self.state_statistics[current_state]['correct'] += 1
                    
                    # Сохраняем результат с полной информацией
                    result = {
                        'index': idx,
                        'price': prices[idx],
                        'prediction': prediction,
                        'actual': actual_outcome,
                        'is_correct': is_correct,
                        'confidence': pred_result['confidence'],
                        'success_rate': self.success_rate,
                        'correct_total': f"{self.correct_predictions}-{self.total_predictions}",
                        'state': current_state,  # Явно включаем состояние
                        'quantile_predictions': pred_result.get('quantile_predictions', {})
                    }
                else:
                    # Если предсказание "не знаю"
                    result = {
                        'index': idx,
                        'price': prices[idx],
                        'prediction': 0,
                        'confidence': pred_result.get('confidence', 0.0),
                        'success_rate': self.success_rate if self.total_predictions > 0 else 0,
                        'state': current_state,  # Явно включаем состояние
                    }
                
                results.append(result)
                
                # Обучаем модель квантильной регрессии на основе этого результата
                if idx % 100 == 0 and idx > min_idx + self.config.prediction_depth:
                    self._update_quantile_models(prices, results)
                
                # Обновляем прогресс-бар
                pbar.update(1)
                if self.total_predictions > 0:
                    pbar.set_postfix({
                        'Predictions': self.total_predictions,
                        'Success Rate': f"{self.success_rate*100:.2f}%"
                    })
            
            # Сохраняем ссылку на массив цен для использования в generate_report
            self.prices = prices
            
            return results
       
    def _update_quantile_models(self, prices, results):
        # Группируем результаты по состояниям
        state_data = defaultdict(lambda: {'X': [], 'y': []})
        
        for r in results:
            if 'state' not in r or r['state'] is None:
                continue
                    
            idx = r['index']
            state = r['state']
            
            # Пропускаем точки, для которых нет данных для проверки
            if idx + self.config.prediction_depth >= len(prices):
                continue
            
            # Собираем признаки
            dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
            features = self._collect_enhanced_features(prices, None, idx, dynamic_threshold)
            
            # Определяем целевое значение
            current_price = prices[idx]
            future_price = prices[idx + self.config.prediction_depth]
            if current_price != 0:
                pct_change = future_price / current_price - 1
                
                state_data[state]['X'].append(features)
                state_data[state]['y'].append(pct_change)
        
        # Отладочный вывод
        print(f"\nModel update: Collected samples for {len(state_data)} states")
        
        # Обучаем модели для каждого состояния
        models_updated = 0
        for state, data in state_data.items():
            if len(data['X']) >= self.config.min_samples_for_regression:
                X = np.array(data['X'])
                y = np.array(data['y'])
                
                model = QuantileRegressionModel(quantiles=self.config.quantiles, alpha=0.1)
                model.fit(X, y)
                self.quantile_models[state] = model
                models_updated += 1
        
        print(f"Updated {models_updated} state models, total models: {len(self.quantile_models)}")
        
        # Обучаем базовую модель на всех данных
        all_X = []
        all_y = []
        
        for data in state_data.values():
            all_X.extend(data['X'])
            all_y.extend(data['y'])
        
        if len(all_X) >= self.config.min_samples_for_regression:
            self.base_quantile_model.fit(np.array(all_X), np.array(all_y))
            print(f"Base model fitted with {len(all_X)} samples")
        else:
            print(f"Not enough samples ({len(all_X)}) for base model")