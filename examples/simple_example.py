"""
Простой пример использования MarkovQuantilePredictor для прогнозирования цен BTC.
"""

import os
import sys

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from markov_quantile_predictor import MarkovPredictor, MarkovQuantilePredictor, PredictorConfig
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp


def main():
    # Создаем директорию для отчетов
    ensure_dir("reports")
    
    # Загружаем данные
    try:
        # Пытаемся загрузить пример данных
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_data.csv')
        prices = load_data(data_path)
    except Exception as e:
        print(f"Не удалось загрузить пример данных: {e}")
        # Генерируем тестовые данные
        np.random.seed(42)
        n_points = 10000
        prices = np.cumsum(np.random.normal(0, 1, n_points)) + 1000
    
    # Создаем конфигурацию предиктора
    config = PredictorConfig(
        window_size=750,
        prediction_depth=15,
        min_confidence=0.8,
        state_length=4,
        significant_change_pct=0.4,
        use_weighted_window=False,
        quantiles=(0.1, 0.5, 0.9),
        min_samples_for_regression=10,
        confidence_threshold=0.75,
        max_coverage=0.05
    )
    
    print("Конфигурация предиктора:")
    print(config)
    
    # Создаем и запускаем марковский предиктор
    print("\nЗапуск марковского предиктора...")
    markov_predictor = MarkovPredictor(config)
    markov_results = markov_predictor.run_on_data(prices)
    
    # Создаем и запускаем гибридный предиктор
    print("\nЗапуск гибридного предиктора (марковский + квантильная регрессия)...")
    hybrid_predictor = MarkovQuantilePredictor(config)
    hybrid_results = hybrid_predictor.run_on_data(prices)
    
    # Визуализируем результаты
    timestamp = get_timestamp()
    
    # Марковский предиктор
    print("\nРезультаты марковского предиктора:")
    print(f"- Всего предсказаний: {markov_predictor.total_predictions}")
    print(f"- Правильных предсказаний: {markov_predictor.correct_predictions}")
    print(f"- Успешность: {markov_predictor.success_rate * 100:.2f}%")
    
    markov_save_path = f"reports/markov_{timestamp}.png"
    markov_predictor.visualize_results(prices, markov_results, markov_save_path)
    
    # Гибридный предиктор
    print("\nРезультаты гибридного предиктора:")
    print(f"- Всего предсказаний: {hybrid_predictor.total_predictions}")
    print(f"- Правильных предсказаний: {hybrid_predictor.correct_predictions}")
    print(f"- Успешность: {hybrid_predictor.success_rate * 100:.2f}%")
    
    hybrid_save_path = f"reports/hybrid_{timestamp}.png"
    hybrid_predictor.visualize_results(prices, hybrid_results, hybrid_save_path)
    
    # Генерируем отчеты
    markov_report_path = f"reports/markov_report_{timestamp}.md"
    markov_predictor.generate_report(markov_results, markov_report_path)
    print(f"Отчет марковского предиктора сохранен в {markov_report_path}")
    
    hybrid_report_path = f"reports/hybrid_report_{timestamp}.md"
    hybrid_predictor.generate_report(hybrid_results, hybrid_report_path)
    print(f"Отчет гибридного предиктора сохранен в {hybrid_report_path}")
    
    # Сравниваем результаты
    print("\nСравнение результатов:")
    markov_success = markov_predictor.success_rate * 100
    hybrid_success = hybrid_predictor.success_rate * 100
    
    print(f"- Марковский предиктор: {markov_success:.2f}%")
    print(f"- Гибридный предиктор: {hybrid_success:.2f}%")
    print(f"- Разница: {hybrid_success - markov_success:.2f}%")
    
    # Создаем график сравнения
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Марковский предиктор', 'Гибридный предиктор'],
                   [markov_success, hybrid_success],
                   color=['blue', 'green'], alpha=0.7)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.2f}%", ha='center', fontsize=12)
    
    plt.title('Сравнение успешности предикторов', fontsize=14)
    plt.ylabel('Успешность (%)', fontsize=12)
    plt.ylim([0, max(markov_success, hybrid_success) * 1.15])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    comparison_save_path = f"reports/comparison_{timestamp}.png"
    plt.savefig(comparison_save_path)
    plt.show()
    
    print(f"График сравнения сохранен в {comparison_save_path}")
    print("\nАнализ завершен.")


if __name__ == "__main__":
    main()