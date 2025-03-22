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
    
class EnhancedHybridPredictor(MarkovQuantilePredictor):
    """
    Улучшенный гибридный предиктор, использующий характеристики состояний
    как входные данные для квантильной регрессии (Вариант 3).
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.feature_extractors = self._init_feature_extractors()
    
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
        """
        Делает предсказание для точки с использованием гибридной модели
        
        Параметры:
        prices (numpy.array): массив цен
        volumes (numpy.array, optional): массив объемов торгов
        idx (int): индекс точки
        max_predictions (int, optional): максимальное количество предсказаний
        current_predictions (int): текущее количество предсказаний
        
        Возвращает:
        dict: результат предсказания
        """
        # Базовые проверки (как в родительском классе)
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
        
        # Собираем все признаки для гибридной модели
        features = self._collect_enhanced_features(prices, volumes, idx, dynamic_threshold)
        
        # Если у нас есть обученная модель для текущего состояния, используем ее
        if current_state in self.quantile_models:
            model = self.quantile_models[current_state]
            predictions = model.predict_single(features)
        else:
            # Если модели нет, используем базовую модель (или None)
            if self.base_quantile_model.is_fitted:
                predictions = self.base_quantile_model.predict_single(features)
            else:
                return {'prediction': 0, 'confidence': 0.0}
        
        # Если не удалось получить предсказания, возвращаем пустой результат
        if predictions is None:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Определяем направление и уверенность на основе квантилей
        median = predictions[0.5]  # медиана (50% квантиль)
        lower = predictions[0.1]   # нижний квантиль (10%)
        upper = predictions[0.9]   # верхний квантиль (90%)
        
        # Определяем направление
        prediction = 0
        if median > dynamic_threshold:
            prediction = 1  # рост
        elif median < -dynamic_threshold:
            prediction = 2  # падение
        
        # Рассчитываем уверенность на основе распределения квантилей
        if prediction == 1:  # Рост
            # Уверенность тем выше, чем дальше нижний квантиль от нуля
            confidence = min(1.0, max(0, lower / dynamic_threshold + 0.5))
        elif prediction == 2:  # Падение
            # Уверенность тем выше, чем дальше верхний квантиль от нуля (в отрицательную сторону)
            confidence = min(1.0, max(0, -upper / dynamic_threshold + 0.5))
        else:
            confidence = 0.0
        
        # Применяем фильтр уверенности
        if confidence < self.config.confidence_threshold:
            prediction = 0
            confidence = 0.0
        
        # Формируем итоговый результат
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'state': current_state,
            'quantile_predictions': {q: float(val) for q, val in predictions.items()},
            'features': features.tolist()  # для отладки
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
                # Делаем предсказание с новой гибридной моделью
                pred_result = self.predict_at_point(prices, volumes, idx, max_predictions, current_predictions)
                prediction = pred_result['prediction']
                
                # Если предсказание не "не знаю", проверяем результат
                if prediction != 0:
                    current_predictions += 1
                    dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
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
                    state = pred_result['state']
                    self.state_statistics[state]['total'] += 1
                    if is_correct:
                        self.state_statistics[state]['correct'] += 1
                    
                    # Сохраняем результат
                    result = {
                        'index': idx,
                        'price': prices[idx],
                        'prediction': prediction,
                        'actual': actual_outcome,
                        'is_correct': is_correct,
                        'confidence': pred_result['confidence'],
                        'success_rate': self.success_rate,
                        'correct_total': f"{self.correct_predictions}-{self.total_predictions}",
                        'state': pred_result['state'],
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
                        'state': pred_result.get('state'),
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
        
        return results
    
    def _update_quantile_models(self, prices, results):
        """Обновляет модели квантильной регрессии на основе накопленных данных"""
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
        
        # Обучаем модели для каждого состояния
        for state, data in state_data.items():
            if len(data['X']) >= self.config.min_samples_for_regression:
                X = np.array(data['X'])
                y = np.array(data['y'])
                
                model = QuantileRegressionModel(quantiles=self.config.quantiles, alpha=0.1)
                model.fit(X, y)
                self.quantile_models[state] = model
        
        # Обучаем базовую модель на всех данных
        all_X = []
        all_y = []
        
        for data in state_data.values():
            all_X.extend(data['X'])
            all_y.extend(data['y'])
        
        if len(all_X) >= self.config.min_samples_for_regression:
            self.base_quantile_model.fit(np.array(all_X), np.array(all_y))