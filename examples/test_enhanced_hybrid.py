"""
Тестирование гибридной модели (Вариант 3) с корректным обнаружением плато.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import types

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markov_quantile_predictor import PredictorConfig
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp
from markov_quantile_predictor.models.hybrid_predictor import EnhancedHybridPredictor

# Функция запуска с обнаружением плато
def run_with_plateau_detection(self, prices, volumes=None, plateau_window=1000, min_predictions=50):
    """
    Запускает модель с обнаружением плато в метрике успешности
    
    Параметры:
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов
    plateau_window (int): окно для обнаружения плато (количество точек)
    min_predictions (int): минимальное количество предсказаний для останова
    
    Возвращает:
    list: результаты предсказаний
    """
    # Предварительно вычисляем изменения цен
    self._precompute_changes(prices)
    
    results = []
    
    # Начинаем с точки, где у нас достаточно данных
    min_idx = max(self.config.window_size, self.config.state_length)
    
    # Вычисляем максимальное количество предсказаний
    max_predictions = int(len(prices) * self.config.max_coverage)
    current_predictions = 0
    
    # Для отслеживания плато
    success_rate_history = []
    last_prediction_idx = None
    plateau_start_idx = None
    
    print("Начало обработки данных...")
    
    # Проходим по всем точкам
    for idx in range(min_idx, len(prices) - self.config.prediction_depth):
        # Делаем предсказание
        pred_result = self.predict_at_point(prices, volumes, idx, max_predictions, current_predictions)
        prediction = pred_result['prediction']
        
        # Получаем состояние для этой точки
        dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
        current_state = self._get_state(prices, idx, dynamic_threshold)
        
        # Если предсказание не "не знаю", проверяем результат
        if prediction != 0:
            current_predictions += 1
            actual_outcome = self._determine_outcome(prices, idx, dynamic_threshold)
            
            # Пропускаем проверку, если результат незначительное изменение
            if actual_outcome is None or actual_outcome == 0:
                continue
            
            is_correct = (prediction == actual_outcome)
            
            # Обновляем статистику
            self.total_predictions += 1
            if is_correct:
                self.correct_predictions += 1
            
            # Обновляем успешность
            self.success_rate = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
            success_rate_history.append(self.success_rate)
            
            # Запоминаем индекс последнего предсказания
            last_prediction_idx = idx
            
            # Сохраняем статистику для этой точки
            self.point_statistics[idx] = {
                'correct': self.correct_predictions,
                'total': self.total_predictions,
                'success_rate': self.success_rate
            }
            
            # Обновляем статистику по этому состоянию
            if current_state:
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
                'state': current_state,
                'quantile_predictions': pred_result.get('quantile_predictions', {})
            }
            
            # Проверка плато в успешности
            if self.total_predictions >= min_predictions:
                # Отслеживаем начало плато
                if plateau_start_idx is None and idx - last_prediction_idx > plateau_window:
                    # Если обработали plateau_window точек с последнего предсказания,
                    # но метрика не изменилась, начинаем отсчет плато
                    plateau_start_idx = idx
                    print(f"\nВозможное плато обнаружено на индексе {idx}, успешность: {self.success_rate * 100:.2f}%")
                
                # Если находимся в плато и обработано еще plateau_window точек
                if plateau_start_idx is not None and idx - plateau_start_idx > plateau_window:
                    print(f"\nПлато подтверждено. Успешность стабилизировалась на {self.success_rate * 100:.2f}%")
                    print(f"Обработано {idx - plateau_start_idx} точек без изменения метрики.")
                    print("Останавливаем обработку данных.")
                    break
        else:
            # Если предсказание "не знаю"
            result = {
                'index': idx,
                'price': prices[idx],
                'prediction': 0,
                'confidence': pred_result.get('confidence', 0.0),
                'success_rate': self.success_rate if self.total_predictions > 0 else 0,
                'state': current_state,
            }
        
        results.append(result)
        
        # Обучаем модель квантильной регрессии на основе этого результата
        if idx % 100 == 0 and idx > min_idx + self.config.prediction_depth:
            self._update_quantile_models(prices, results)
            
            # Вывод текущего прогресса
            progress_pct = (idx - min_idx) * 100 / (len(prices) - min_idx - self.config.prediction_depth)
            print(f"\rProcessing: {progress_pct:5.1f}% | Predictions={self.total_predictions}, Success Rate={self.success_rate*100:.2f}%", end="")
    
    print("\n")  # Новая строка после прогресс-бара
    self.prices = prices  # Сохраняем для отчета
    
    return results


def main():
    # Создаем директорию для отчетов
    ensure_dir("reports")
    
    # Загружаем данные
    try:
        # Пытаемся загрузить данные BTC
        data_path = os.path.join(os.path.dirname(__file__), '../data/BTC_price_data.csv')
        df = pd.read_csv(data_path)
        
        # Выделяем цены и объемы (если доступны)
        prices = df['price'].values if 'price' in df.columns else df.iloc[:, 0].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
    except Exception as e:
        print(f"Не удалось загрузить данные: {e}")
        # Генерируем тестовые данные
        np.random.seed(42)
        n_points = 10000
        prices = np.cumsum(np.random.normal(0, 1, n_points)) + 1000
        volumes = None
    
    # Оптимальная конфигурация для достижения ~58% Success Rate
    config = PredictorConfig(
        window_size=750,
        prediction_depth=15,
        min_confidence=0.6,
        state_length=4,
        significant_change_pct=0.01,  # Уменьшенный порог значимого изменения
        use_weighted_window=False,
        quantiles=(0.1, 0.5, 0.9),    # Упрощенный набор квантилей
        min_samples_for_regression=3,  # Минимальное количество образцов
        confidence_threshold=0.55,     # Начальный порог уверенности
        max_coverage=0.1               # Увеличенное покрытие
    )

    print("Конфигурация гибридного предиктора:")
    print(config)
    
    # Создаем гибридную модель
    print("\nСоздание гибридной модели...")
    hybrid_predictor = EnhancedHybridPredictor(config)
    
    # Предварительное обучение на 40% данных
    training_data_size = int(len(prices) * 0.4)
    training_prices = prices[:training_data_size]
    training_volumes = volumes[:training_data_size] if volumes is not None else None
    
    print(f"\nПредварительное обучение на {training_data_size} точках...")
    hybrid_predictor.run_on_data(training_prices, training_volumes, verbose=True)
    
    print(f"\nПредварительное обучение завершено:")
    print(f"- Количество обученных моделей состояний: {len(hybrid_predictor.quantile_models)}")
    print(f"- Базовая модель обучена: {hybrid_predictor.base_quantile_model.is_fitted}")
    print(f"- Текущая успешность: {hybrid_predictor.success_rate * 100:.2f}%")
    
    # Настройка для основного запуска
    print("\nНастройка конфигурации для основного запуска...")
    hybrid_predictor.config.confidence_threshold = 0.58  # Оптимальный порог
    
    # Сбрасываем статистику
    hybrid_predictor.total_predictions = 0
    hybrid_predictor.correct_predictions = 0
    hybrid_predictor.success_rate = 0.0
    hybrid_predictor.point_statistics = {}
    
    # Добавляем функцию обнаружения плато
    hybrid_predictor.run_with_plateau_detection = types.MethodType(run_with_plateau_detection, hybrid_predictor)
    
    # Запускаем с обнаружением плато
    print("\nЗапуск основного тестирования с обнаружением плато...")
    hybrid_results = hybrid_predictor.run_with_plateau_detection(
        prices, 
        volumes,
        plateau_window=1000,  # Окно 1000 точек для обнаружения плато
        min_predictions=50    # Минимум 50 предсказаний для статистической значимости
    )
    
    # Визуализируем результаты
    timestamp = get_timestamp()
    
    # Результаты гибридной модели
    print("\nРезультаты гибридной модели:")
    print(f"- Всего предсказаний: {hybrid_predictor.total_predictions}")
    print(f"- Правильных предсказаний: {hybrid_predictor.correct_predictions}")
    print(f"- Успешность: {hybrid_predictor.success_rate * 100:.2f}%")
    
    hybrid_save_path = f"reports/enhanced_hybrid_{timestamp}.png"
    hybrid_predictor.visualize_results(prices, hybrid_results, hybrid_save_path)
    
    # Генерируем отчет
    hybrid_report_path = f"reports/enhanced_hybrid_report_{timestamp}.md"
    hybrid_predictor.generate_report(hybrid_results, hybrid_report_path, prices)
    print(f"Отчет гибридной модели сохранен в {hybrid_report_path}")
    
    print("\nАнализ завершен.")

if __name__ == "__main__":
    main()