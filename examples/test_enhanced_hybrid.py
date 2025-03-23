"""
Тестирование гибридной модели (Вариант 3).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markov_quantile_predictor import PredictorConfig
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp
from markov_quantile_predictor.models.hybrid_predictor import EnhancedHybridPredictor  # наш новый класс

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
    
    # Создаем конфигурацию предиктора с уменьшенным набором квантилей
    config = PredictorConfig(
        window_size=750,
        prediction_depth=15,
        min_confidence=0.6,
        state_length=4,
        significant_change_pct=0.4,
        use_weighted_window=False,
        # quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        quantiles=(0.1, 0.5, 0.9),
        min_samples_for_regression=10,
        confidence_threshold=0.5,
        max_coverage=0.05
    )

    print("Конфигурация гибридного предиктора:")
    print(config)
    
    # Создаем и запускаем гибридную модель
    print("\nЗапуск гибридной модели (Вариант 3)...")
    hybrid_predictor = EnhancedHybridPredictor(config)
    hybrid_results = hybrid_predictor.run_on_data(prices, volumes)
    
    # Визуализируем результаты
    timestamp = get_timestamp()
    
    # Гибридная модель
    print("\nРезультаты гибридной модели:")
    print(f"- Всего предсказаний: {hybrid_predictor.total_predictions}")
    print(f"- Правильных предсказаний: {hybrid_predictor.correct_predictions}")
    print(f"- Успешность: {hybrid_predictor.success_rate * 100:.2f}%")
    
    hybrid_save_path = f"reports/enhanced_hybrid_{timestamp}.png"
    hybrid_predictor.visualize_results(prices, hybrid_results, hybrid_save_path)
    
    # Генерируем отчет
    hybrid_report_path = f"reports/enhanced_hybrid_report_{timestamp}.md"
    hybrid_predictor.generate_report(hybrid_results, hybrid_report_path, prices)  # Передаем массив цен    
    print(f"Отчет гибридной модели сохранен в {hybrid_report_path}")
    
    print("\nАнализ завершен.")

if __name__ == "__main__":
    main()