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
    
    def predict_at_point(self, prices, idx, max_predictions=None, current_predictions=0):
        """
        Делает предсказание для точки, используя сначала марковский предиктор,
        а затем квантильную регрессию
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): индекс точки
        max_predictions (int, optional): максимальное количество предсказаний
        current_predictions (int): текущее количество предсказаний
        
        Возвращает:
        dict: результат предсказания
        """
        # Шаг 1: Получаем предсказание от марковского предиктора
        markov_result = super().predict_at_point(prices, idx, max_predictions, current_predictions)
        
        # Если марковский предиктор не дал предсказания, возвращаем его результат
        if markov_result['prediction'] == 0:
            return markov_result
        
        # Шаг 2: Получаем данные для квантильной регрессии
        dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
        X, y, current_features = self._collect_state_samples(prices, idx, dynamic_threshold)
        
        current_state = markov_result['state']
        
        # Если недостаточно данных для квантильной регрессии, возвращаем предсказание марковского предиктора
        if X is None or len(X) < self.config.min_samples_for_regression:
            return markov_result
        
        # Шаг 3: Обучаем или используем модель квантильной регрессии для текущего состояния
        if current_state not in self.quantile_models:
            # Создаем и обучаем новую модель
            model = QuantileRegressionModel(quantiles=self.config.quantiles, alpha=0.1)
            model.fit(X, y)
            self.quantile_models[current_state] = model
        else:
            # Используем существующую модель
            model = self.quantile_models[current_state]
        
        # Шаг 4: Делаем предсказание с помощью квантильной регрессии
        quantile_predictions = model.predict_single(current_features)
        
        # Если не удалось сделать предсказание, возвращаем результат марковского предиктора
        if quantile_predictions is None:
            return markov_result
        
        # Шаг 5: Интегрируем результаты обоих моделей
        
        # Получаем медианное предсказание (квантиль 0.5)
        median_prediction = quantile_predictions[0.5]
        
        # Определяем направление движения на основе квантильной регрессии
        quantile_direction = 0
        if median_prediction > dynamic_threshold:
            quantile_direction = 1  # Рост
        elif median_prediction < -dynamic_threshold:
            quantile_direction = 2  # Падение
        
        # Проверяем, согласуются ли предсказания обеих моделей
        if quantile_direction == markov_result['prediction']:
            # Если предсказания согласуются, увеличиваем уверенность
            confidence = min(1.0, markov_result['confidence'] * 1.1)
        else:
            # Если предсказания не согласуются, снижаем уверенность
            confidence = markov_result['confidence'] * 0.8
            
            # Если уверенность стала ниже порога, отменяем предсказание
            if confidence < self.config.confidence_threshold:
                return {
                    'prediction': 0,
                    'confidence': 0.0,
                    'state': current_state,
                    'state_occurrences': markov_result.get('state_occurrences', 0)
                }
        
        # Формируем итоговый результат, добавляя информацию от квантильной регрессии
        result = {
            'prediction': markov_result['prediction'],
            'confidence': confidence,
            'state': current_state,
            'state_occurrences': markov_result.get('state_occurrences', 0),
            'quantile_predictions': {q: float(val) for q, val in quantile_predictions.items()},
            'up_prob': markov_result.get('up_prob', 0),
            'down_prob': markov_result.get('down_prob', 0)
        }
        
        return result
    
    def visualize_results(self, prices, results, save_path=None):
        """
        Визуализирует результаты предсказаний, включая информацию от квантильной регрессии
        
        Параметры:
        prices (numpy.array): массив цен
        results (list): результаты предсказаний
        save_path (str): путь для сохранения графиков
        """
        # Сначала используем метод визуализации родительского класса
        super().visualize_results(prices, results, save_path)
        
        # Дополнительно создаем график с информацией от квантильной регрессии
        quantile_results = [r for r in results if 'quantile_predictions' in r]
        
        if not quantile_results:
            return  # Нет данных для визуализации
        
        # Создаем новый график
        plt.figure(figsize=(15, 8))
        
        # График цен
        plt.plot(prices, color='blue', alpha=0.5, label='Цена')
        
        # Собираем данные для отображения квантилей
        indices = []
        medians = []
        lower_bounds = []
        upper_bounds = []
        
        for r in quantile_results:
            idx = r['index']
            indices.append(idx)
            
            # Получаем предсказания квантилей
            q_preds = r['quantile_predictions']
            
            # Преобразуем из процентного изменения в абсолютное значение цены
            price = prices[idx]
            medians.append(price * (1 + q_preds[0.5]))
            lower_bounds.append(price * (1 + q_preds[0.1]))
            upper_bounds.append(price * (1 + q_preds[0.9]))
        
        # Отображаем медианные предсказания
        plt.scatter(indices, medians, color='green', s=30, alpha=0.7, label='Медианное предсказание')
        
        # Отображаем интервалы предсказаний
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]], [lower_bounds[i], upper_bounds[i]], 
                     color='green', alpha=0.3)
        
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
    
    def generate_report(self, results, save_path=None):
        """
        Генерирует отчет о результатах предсказаний, включая информацию от квантильной регрессии
        
        Параметры:
        results (list): результаты предсказаний
        save_path (str): путь для сохранения отчета
        
        Возвращает:
        str: текст отчета
        """
        # Получаем базовый отчет от родительского класса
        base_report = super().generate_report(results, None)
        
        # Собираем статистику по квантильным предсказаниям
        quantile_results = [r for r in results if 'quantile_predictions' in r]
        
        if not quantile_results:
            # Если нет данных от квантильной регрессии, возвращаем базовый отчет
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(base_report)
            return base_report
        
        # Анализируем результаты квантильных предсказаний
        
        # 1. Точность квантильных предсказаний
        # Считаем сколько раз фактическое значение попало в интервал [q10, q90]
        actual_in_interval = 0
        
        for r in quantile_results:
            idx = r['index']
            if idx + self.config.prediction_depth >= len(prices):
                continue
                
            # Фактическое изменение
            actual_change = prices[idx + self.config.prediction_depth] / prices[idx] - 1
            
            # Предсказанные квантили
            q10 = r['quantile_predictions'][0.1]
            q90 = r['quantile_predictions'][0.9]
            
            # Проверяем, попало ли фактическое изменение в интервал
            if q10 <= actual_change <= q90:
                actual_in_interval += 1
        
        # Вычисляем процент попаданий в интервал
        interval_coverage = actual_in_interval / len(quantile_results) * 100 if quantile_results else 0
        
        # 2. Средние предсказанные изменения для разных квантилей
        mean_q10 = np.mean([r['quantile_predictions'][0.1] * 100 for r in quantile_results])
        mean_q50 = np.mean([r['quantile_predictions'][0.5] * 100 for r in quantile_results])
        mean_q90 = np.mean([r['quantile_predictions'][0.9] * 100 for r in quantile_results])
        
        # 3. Ширина предсказанных интервалов
        mean_interval_width = np.mean([(r['quantile_predictions'][0.9] - r['quantile_predictions'][0.1]) * 100 
                                       for r in quantile_results])
        
        # Формируем дополнительный отчет
        quantile_report = f"""
## Статистика квантильной регрессии

### Общая информация
- Количество предсказаний с квантильной регрессией: {len(quantile_results)}
- Процент попаданий фактического значения в интервал [10%, 90%]: {interval_coverage:.2f}%

### Средние предсказанные изменения
- Средний нижний квантиль (10%): {mean_q10:.2f}%
- Средний медианный квантиль (50%): {mean_q50:.2f}%
- Средний верхний квантиль (90%): {mean_q90:.2f}%
- Средняя ширина интервала [10%, 90%]: {mean_interval_width:.2f}%

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