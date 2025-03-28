"""
Модуль с реализацией марковского предиктора.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from ..predictor_config import PredictorConfig


class MarkovPredictor:
    """
    Предиктор на основе марковского процесса со скользящим окном.
    """
    
    def __init__(self, config=None):
        """
        Инициализация предиктора
        
        Параметры:
        config (PredictorConfig): конфигурация параметров предиктора
        """
        # Используем конфигурацию по умолчанию, если не передана
        self.config = config if config else PredictorConfig()
        
        # Статистика
        self.total_predictions = 0
        self.correct_predictions = 0
        self.success_rate = 0.0
        
        # История статистики по точкам
        self.point_statistics = {}
        
        # Для детального анализа
        self.state_statistics = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # Предварительно вычисленные изменения цен
        self.price_changes = None
    
    def _precompute_changes(self, prices):
        """
        Предварительно вычисляет относительные изменения цен
        
        Параметры:
        prices (numpy.array): массив цен
        """
        if self.price_changes is None:
            self.price_changes = np.zeros(len(prices) - 1)
            for i in range(len(prices) - 1):
                if prices[i] != 0:
                    self.price_changes[i] = (prices[i + 1] - prices[i]) / prices[i]
                else:
                    self.price_changes[i] = 0
    
    def _get_weights(self, window_length, prices=None, start_idx=None):
        """
        Генерирует веса для точек в окне
        
        Параметры:
        window_length (int): длина окна
        prices (numpy.array, optional): массив цен
        start_idx (int, optional): начальный индекс в массиве цен
        
        Возвращает:
        numpy.array: массив весов
        """
        if not self.config.use_weighted_window:
            return np.ones(window_length)
        
        # Базовое экспоненциальное затухание - новые точки имеют больший вес
        weights = np.array([self.config.weight_decay ** i for i in range(window_length - 1, -1, -1)])
        
        # Адаптивное взвешивание - усиливаем значимость недавних точек
        if self.config.adaptive_weighting:
            # Определяем точку перегиба (середина окна)
            pivot = window_length // 2
            
            # Применяем усиленное взвешивание для последних точек
            for i in range(pivot, window_length):
                boost_factor = 1.0 + (i - pivot) / (window_length - pivot) * (self.config.recency_boost - 1.0)
                weights[i] *= boost_factor
        
        # Взвешивание по волатильности - точки с высокой волатильностью имеют меньший вес
        if self.config.volatility_weighting and prices is not None and start_idx is not None:
            volatility = []
            for i in range(window_length - 1):
                idx = start_idx + i
                if idx + 1 < len(prices):
                    # Вычисляем абсолютное процентное изменение
                    pct_change = abs((prices[idx + 1] - prices[idx]) / prices[idx]) if prices[idx] != 0 else 0
                    volatility.append(pct_change)
                else:
                    volatility.append(0)
            
            # Добавляем последнюю точку
            volatility.append(0)
            
            # Нормализуем волатильность
            max_vol = max(volatility) if max(volatility) > 0 else 1
            volatility = [v / max_vol for v in volatility]
            
            # Инвертируем веса волатильности (более стабильные периоды имеют больший вес)
            vol_weights = [1 - 0.5 * v for v in volatility]
            
            # Применяем веса волатильности
            weights = weights * np.array(vol_weights)
        
        # Нормализуем, чтобы сумма была равна window_length
        weights = weights * window_length / weights.sum()
        
        return weights
    
    # Как в успешной версии
    def _calculate_dynamic_threshold(self, prices, idx):
        """
        Вычисляет динамический порог для определения значимого изменения цены
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        
        Возвращает:
        float: динамический порог
        """
        start_idx = max(0, idx - self.config.window_size)
        if start_idx >= len(self.price_changes):
            return self.config.significant_change_pct
        
        changes = self.price_changes[start_idx:idx-1]
        
        # Используем 75-й процентиль абсолютных изменений или заданный порог, если данных недостаточно
        result = np.percentile(np.abs(changes), 75) if len(changes) > 0 else self.config.significant_change_pct
        
        if idx % 1000 == 0:
            print(f"DEBUG: dynamic_threshold at idx={idx}: {result}, config threshold: {self.config.significant_change_pct}")
        
        return result
        
    def _determine_movement(self, current_price, next_price, dynamic_threshold):
        """
        Определяет направление движения с учетом порога значимого изменения
        
        Параметры:
        current_price (float): текущая цена
        next_price (float): следующая цена
        dynamic_threshold (float): динамический порог для определения значимого изменения
        
        Возвращает:
        int: 1 = значимый рост, 2 = значимое падение, 0 = незначительное изменение
        """
        # Вычисляем процентное изменение
        if current_price == 0:
            return 0  # Избегаем деления на ноль
        
        pct_change = (next_price - current_price) / current_price
        
        # Применяем порог значимого изменения
        if pct_change > dynamic_threshold:
            return 1  # Значимый рост
        elif pct_change < -dynamic_threshold:
            return 2  # Значимое падение
        else:
            return 0  # Незначительное изменение
    
    def _get_state(self, prices, idx, dynamic_threshold):
        """
        Определяет текущее состояние рынка
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        dynamic_threshold (float): динамический порог
        
        Возвращает:
        tuple: состояние рынка (последовательность движений)
        """
        # Нужно иметь как минимум state_length + 1 точек для определения состояния
        if idx < self.config.state_length:
            return None  # Недостаточно данных для определения состояния
        
        # Определяем последние state_length движений
        movements = []
        for i in range(idx - self.config.state_length, idx):
            movement = self._determine_movement(prices[i], prices[i+1], dynamic_threshold)
            # Для незначительных изменений (0) будем считать их как продолжение предыдущего движения
            # Если это первое движение в состоянии, считаем его нейтральным (1)
            if movement == 0:
                if len(movements) > 0:
                    movement = movements[-1]  # Продолжаем предыдущее движение
                else:
                    movement = 1  # По умолчанию нейтральное движение
            movements.append(movement)
        
        return tuple(movements)
    
    def _determine_outcome(self, prices, idx, dynamic_threshold):
        """
        Определяет фактический исход через prediction_depth точек
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        dynamic_threshold (float): динамический порог
        
        Возвращает:
        int: 1 = рост, 2 = падение, 0 = незначительное изменение
        """
        if idx + self.config.prediction_depth >= len(prices):
            return None  # Нет данных для проверки
        
        current_price = prices[idx]
        future_price = prices[idx + self.config.prediction_depth]
        
        # Используем тот же порог значимого изменения
        return self._determine_movement(current_price, future_price, dynamic_threshold)
    
    def _get_probabilities(self, prices, idx):
        """
        Вычисляет базовые вероятности направления движения на основе исторических данных
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): текущий индекс
        
        Возвращает:
        dict: словарь с вероятностями направлений
        """
        start_idx = max(0, idx - self.config.window_size)
        if start_idx >= len(self.price_changes):
            return {'up_prob': 0.0, 'down_prob': 0.0}
        
        changes = self.price_changes[start_idx:idx-1]
        if len(changes) == 0:
            return {'up_prob': 0.0, 'down_prob': 0.0}
        
        # Вычисляем долю положительных и отрицательных изменений
        up_prob = np.mean(changes > 0)
        down_prob = np.mean(changes < 0)
        
        return {'up_prob': up_prob, 'down_prob': down_prob}
    
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
        
        # Определяем направление и уверенность на основе квантилей
        median = predictions[0.5]  # медиана (50% квантиль)
        lower = predictions[0.1]   # нижний квантиль (10%)
        upper = predictions[0.9]   # верхний квантиль (90%)
        
        # Дополнительная отладочная информация
        if idx % 1000 == 0:
            print(f"  Median: {median:.6f}, Lower: {lower:.6f}, Upper: {upper:.6f}")
        
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
        
        # Отладочная информация о решении
        if idx % 1000 == 0:
            print(f"  Decision: prediction={prediction}, confidence={confidence:.4f}, threshold={self.config.confidence_threshold}")
        
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
    
    def run_on_data(self, prices, verbose=True):
        """
        Последовательно проходит по данным, делая предсказания и проверяя их
        
        Параметры:
        prices (numpy.array): массив цен
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
                # Делаем предсказание, анализируя предыдущие window_size точек
                pred_result = self.predict_at_point(prices, idx, max_predictions, current_predictions)
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
                        'state_occurrences': pred_result.get('state_occurrences', 0)
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
                        'state_occurrences': pred_result.get('state_occurrences', 0)
                    }
                
                results.append(result)
                
                # Обновляем прогресс-бар
                pbar.update(1)
                if self.total_predictions > 0:
                    pbar.set_postfix({
                        'Predictions': self.total_predictions,
                        'Success Rate': f"{self.success_rate*100:.2f}%"
                    })
        
        return results
    
    def visualize_results(self, prices, results, save_path=None):
        """
        Визуализирует результаты предсказаний
        
        Параметры:
        prices (numpy.array): массив цен
        results (list): результаты предсказаний
        save_path (str): путь для сохранения графиков
        """
        # Создаем графики
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # График цен
        ax1.plot(prices, color='blue', alpha=0.7, label='Цена')
        
        # Выделяем обучающий участок
        ax1.axvspan(0, self.config.window_size, color='lightgray', alpha=0.3, label='Начальное окно')
        
        # Отмечаем предсказания
        correct_up_indices = []
        correct_up_prices = []
        correct_down_indices = []
        correct_down_prices = []
        
        wrong_up_indices = []
        wrong_up_prices = []
        wrong_down_indices = []
        wrong_down_prices = []
        
        for r in results:
            idx = r['index']
            price = r['price']
            
            if 'is_correct' in r:
                if r['prediction'] == 1:  # Up
                    if r['is_correct']:
                        correct_up_indices.append(idx)
                        correct_up_prices.append(price)
                    else:
                        wrong_up_indices.append(idx)
                        wrong_up_prices.append(price)
                elif r['prediction'] == 2:  # Down
                    if r['is_correct']:
                        correct_down_indices.append(idx)
                        correct_down_prices.append(price)
                    else:
                        wrong_down_indices.append(idx)
                        wrong_down_prices.append(price)
        
        # Отмечаем предсказания на графике
        if correct_up_indices:
            ax1.scatter(correct_up_indices, correct_up_prices, color='green', marker='^', s=50, alpha=0.7, 
                      label='Верно (Рост)')
        if correct_down_indices:
            ax1.scatter(correct_down_indices, correct_down_prices, color='green', marker='v', s=50, alpha=0.7, 
                      label='Верно (Падение)')
        if wrong_up_indices:
            ax1.scatter(wrong_up_indices, wrong_up_prices, color='red', marker='^', s=50, alpha=0.7, 
                      label='Неверно (Рост)')
        if wrong_down_indices:
            ax1.scatter(wrong_down_indices, wrong_down_prices, color='red', marker='v', s=50, alpha=0.7, 
                      label='Неверно (Падение)')
        
        ax1.set_title('Цена и предсказания')
        ax1.set_ylabel('Цена')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # График успешности предсказаний
        success_indices = []
        success_rates = []
        
        for idx, stats in sorted(self.point_statistics.items()):
            success_indices.append(idx)
            success_rates.append(stats['success_rate'] * 100)
        
        if success_indices:
            ax2.plot(success_indices, success_rates, 'g-', linewidth=2)
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7)
            ax2.set_title('Динамика успешности предсказаний')
            ax2.set_xlabel('Индекс')
            ax2.set_ylabel('Успешность (%)')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график, если указан путь
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        
        # График распределения состояний и их успешности
        if self.state_statistics:
            # Берем топ-10 самых распространенных состояний
            top_states = sorted(
                [(state, stats['total']) for state, stats in self.state_statistics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            state_labels = [str(state) for state, _ in top_states]
            state_counts = [count for _, count in top_states]
            state_success_rates = [
                self.state_statistics[state]['correct'] / self.state_statistics[state]['total'] * 100
                if self.state_statistics[state]['total'] > 0 else 0
                for state, _ in top_states
            ]
            
            # Создаем график для топ-10 состояний
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # График количества состояний
            ax1.bar(state_labels, state_counts, color='blue', alpha=0.7)
            ax1.set_title('Частота встречаемости топ-10 состояний')
            ax1.set_ylabel('Количество')
            ax1.set_xticklabels(state_labels, rotation=45)
            ax1.grid(alpha=0.3, axis='y')
            
            # График успешности по состояниям
            ax2.bar(state_labels, state_success_rates, color='green', alpha=0.7)
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7)
            ax2.set_title('Успешность предсказаний по состояниям')
            ax2.set_ylabel('Успешность (%)')
            ax2.set_xticklabels(state_labels, rotation=45)
            ax2.grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Сохраняем график состояний, если указан путь
            if save_path:
                states_path = save_path.replace('.png', '_states.png')
                plt.savefig(states_path)
            
            plt.show()
    
    def get_state_statistics(self):
        """
        Возвращает статистику по состояниям
        
        Возвращает:
        pandas.DataFrame: статистика по состояниям
        """
        data = []
        for state, stats in self.state_statistics.items():
            success_rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            data.append({
                'state': str(state),
                'total': stats['total'],
                'correct': stats['correct'],
                'success_rate': success_rate
            })
        
        # Проверка на пустой список
        if not data:
            # Возвращаем пустой DataFrame с нужными столбцами
            return pd.DataFrame(columns=['state', 'total', 'correct', 'success_rate'])
        
        df = pd.DataFrame(data)
        return df.sort_values('total', ascending=False)
        
    def generate_report(self, results, save_path=None):
        """
        Генерирует подробный отчет о результатах предсказаний
        
        Параметры:
        results (list): результаты предсказаний
        save_path (str): путь для сохранения отчета
        
        Возвращает:
        str: текст отчета
        """
        # Общая статистика
        total_predictions = self.total_predictions
        correct_predictions = self.correct_predictions
        success_rate = self.success_rate * 100
        
        # Распределение предсказаний
        up_count = sum(1 for r in results if r.get('prediction') == 1)
        down_count = sum(1 for r in results if r.get('prediction') == 2)
        neutral_count = sum(1 for r in results if r.get('prediction') == 0)
        
        # Успешность по типам предсказаний
        up_correct = sum(1 for r in results if r.get('prediction') == 1 and r.get('is_correct', False))
        down_correct = sum(1 for r in results if r.get('prediction') == 2 and r.get('is_correct', False))
        
        up_success_rate = up_correct / up_count * 100 if up_count > 0 else 0
        down_success_rate = down_correct / down_count * 100 if down_count > 0 else 0
        
        # Топ состояния
        state_stats = self.get_state_statistics()
        top_states = state_stats.head(10)
        
        # Формируем отчет
        report = f"""
# Отчет о работе предиктора

## Конфигурация
{str(self.config)}

## Общая статистика
- Всего предсказаний: {total_predictions}
- Правильных предсказаний: {correct_predictions}
- Успешность: {success_rate:.2f}%

## Распределение предсказаний
- Рост: {up_count} ({up_count/len(results)*100:.2f}%)
- Падение: {down_count} ({down_count/len(results)*100:.2f}%)
- Не знаю: {neutral_count} ({neutral_count/len(results)*100:.2f}%)

## Успешность по типам предсказаний
- Успешность предсказаний роста: {up_correct}/{up_count} ({up_success_rate:.2f}%)
- Успешность предсказаний падения: {down_correct}/{down_count} ({down_success_rate:.2f}%)

## Топ-10 состояний по частоте
{top_states.to_markdown(index=False)}

## Покрытие предсказаний
- Общее покрытие: {(up_count + down_count) / len(results) * 100:.2f}%
"""
        
        # Сохраняем отчет, если указан путь
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report