"""
Модификация функции run_on_data для EnhancedHybridPredictor,
добавляющая обнаружение плато метрик и ранний останов.
"""
import numpy as np
from tqdm import tqdm  


def run_on_data_with_early_stopping(self, prices, volumes=None, verbose=True, 
                                    plateau_window_percent=5.0, 
                                    plateau_threshold=0.1,
                                    min_progress_percent=15.0):
    """
    Запускает модель на данных с ранним остановом при обнаружении плато метрик
    
    Параметры:
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов торгов
    verbose (bool): выводить информацию о прогрессе
    plateau_window_percent (float): размер окна в процентах для обнаружения плато
    plateau_threshold (float): пороговое значение изменения метрик для определения плато
    min_progress_percent (float): минимальный процент прогресса перед проверкой плато
    
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
    
    # Для обнаружения плато
    success_rate_history = []
    last_check_idx = min_idx
    
    # Рассчитываем размер окна для проверки плато
    plateau_window_size = int((len(prices) - min_idx - self.config.prediction_depth) * plateau_window_percent / 100)
    
    # Рассчитываем минимальный индекс для проверки плато
    min_progress_idx = min_idx + int((len(prices) - min_idx - self.config.prediction_depth) * min_progress_percent / 100)
    
    print(f"Настройки раннего останова:")
    print(f"- Окно проверки плато: {plateau_window_percent}% ({plateau_window_size} точек)")
    print(f"- Порог плато: {plateau_threshold}%")
    print(f"- Минимальный прогресс перед проверкой: {min_progress_percent}%")
    
    # Проходим по всем точкам
    with tqdm(total=len(prices) - min_idx - self.config.prediction_depth, 
              desc="Processing", disable=not verbose) as pbar:
        for idx in range(min_idx, len(prices) - self.config.prediction_depth):
            # Делаем предсказание, анализируя предыдущие window_size точек
            pred_result = self.predict_at_point(prices, volumes, idx, max_predictions, current_predictions)
            prediction = pred_result['prediction']
            
            # Если предсказание не "не знаю", проверяем результат через prediction_depth точек
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
                
                # Добавляем текущую успешность в историю
                success_rate_history.append(self.success_rate * 100)
                
                # Сохраняем статистику для этой точки
                self.point_statistics[idx] = {
                    'correct': self.correct_predictions,
                    'total': self.total_predictions,
                    'success_rate': self.success_rate
                }
                
                # Обновляем статистику по этому состоянию
                state = pred_result['state']
                if state:  # Проверяем, что состояние определено
                    self.state_statistics[state]['total'] += 1
                    if is_correct:
                        self.state_statistics[state]['correct'] += 1
                
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
                    'state': state,  # Явно включаем состояние
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
                    'state': pred_result.get('state'),  # Явно включаем состояние
                }
            
            results.append(result)
            
            # Обучаем модель квантильной регрессии каждые 100 точек
            if idx % 100 == 0 and idx > min_idx + self.config.prediction_depth:
                self._update_quantile_models(prices, results)
            
            # Обновляем прогресс-бар
            pbar.update(1)
            if self.total_predictions > 0:
                pbar.set_postfix({
                    'Predictions': self.total_predictions,
                    'Success Rate': f"{self.success_rate*100:.2f}%"
                })
            
            # Проверяем плато, если прошли минимальный процент и окно проверки плато
            current_progress_percent = (idx - min_idx) * 100 / (len(prices) - min_idx - self.config.prediction_depth)
            
            # Проверяем, достаточно ли данных и прошли ли минимальный процент
            if idx >= min_progress_idx and (idx - last_check_idx) >= plateau_window_size:
                # Проверяем изменение Success Rate за последнее окно
                if len(success_rate_history) >= 2:
                    # Берем среднюю успешность за первую и вторую половину окна
                    window_size = min(len(success_rate_history), plateau_window_size)
                    mid_point = max(1, window_size // 2)
                    
                    # Находим среднюю успешность для первой и второй половины окна
                    first_half_avg = sum(success_rate_history[-window_size:-mid_point]) / (window_size - mid_point)
                    second_half_avg = sum(success_rate_history[-mid_point:]) / mid_point
                    
                    # Вычисляем абсолютное изменение
                    abs_change = abs(second_half_avg - first_half_avg)
                    
                    # Если изменение меньше порога, считаем что достигли плато
                    if abs_change < plateau_threshold:
                        print(f"\nОбнаружено плато метрик на {current_progress_percent:.1f}% обработки данных:")
                        print(f"- Изменение Success Rate: {abs_change:.3f}% (порог: {plateau_threshold}%)")
                        print(f"- Первая половина окна: {first_half_avg:.2f}%")
                        print(f"- Вторая половина окна: {second_half_avg:.2f}%")
                        print(f"- Текущая успешность: {self.success_rate*100:.2f}%")
                        print(f"- Количество предсказаний: {self.total_predictions}")
                        print("Останавливаем обработку данных.")
                        break
                
                # Обновляем индекс последней проверки
                last_check_idx = idx
    
    # Сохраняем ссылку на массив цен для использования в generate_report
    self.prices = prices
    
    return results