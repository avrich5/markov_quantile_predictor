"""
Пример сравнения различных конфигураций предикторов.
"""

import os
import sys

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from markov_quantile_predictor import MarkovPredictor, MarkovQuantilePredictor, PredictorConfig
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp, visualize_validation_results


def run_model_comparison(data_path, configs, save_dir="reports"):
    """
    Сравнивает различные конфигурации предикторов
    
    Параметры:
    data_path (str): путь к CSV-файлу с данными
    configs (list): список конфигураций для сравнения
    save_dir (str): директория для сохранения отчетов
    
    Возвращает:
    pandas.DataFrame: таблица с результатами сравнения
    """
    # Создаем директорию для отчетов
    ensure_dir(save_dir)
    
    # Загружаем данные
    prices = load_data(data_path)
    
    # Создаем таблицу для результатов
    results = []
    
    # Общая временная метка для группы отчетов
    timestamp = get_timestamp()
    
    # Проходим по всем конфигурациям
    for i, config in enumerate(configs):
        print(f"\nЗапуск конфигурации {i+1}/{len(configs)}:")
        print(config)
        
        # Запускаем марковский предиктор
        print("- Марковский предиктор...")
        markov_predictor = MarkovPredictor(config)
        markov_results = markov_predictor.run_on_data(prices)
        
        markov_save_path = f"{save_dir}/markov_config{i+1}_{timestamp}.png"
        markov_predictor.visualize_results(prices, markov_results, markov_save_path)
        
        markov_report_path = f"{save_dir}/markov_report_config{i+1}_{timestamp}.md"
        markov_predictor.generate_report(markov_results, markov_report_path)
        
        # Запускаем гибридный предиктор
        print("- Гибридный предиктор...")
        hybrid_predictor = MarkovQuantilePredictor(config)
        hybrid_results = hybrid_predictor.run_on_data(prices)
        
        hybrid_save_path = f"{save_dir}/hybrid_config{i+1}_{timestamp}.png"
        hybrid_predictor.visualize_results(prices, hybrid_results, hybrid_save_path)
        
        hybrid_report_path = f"{save_dir}/hybrid_report_config{i+1}_{timestamp}.md"
        hybrid_predictor.generate_report(hybrid_results, hybrid_report_path)
        
        # Добавляем результаты в таблицу
        results.append({
            'Config': f"Config {i+1}",
            'Markov Success Rate (%)': markov_predictor.success_rate * 100,
            'Markov Predictions': markov_predictor.total_predictions,
            'Markov Coverage (%)': markov_predictor.total_predictions / len(prices) * 100,
            'Hybrid Success Rate (%)': hybrid_predictor.success_rate * 100,
            'Hybrid Predictions': hybrid_predictor.total_predictions,
            'Hybrid Coverage (%)': hybrid_predictor.total_predictions / len(prices) * 100,
            'Improvement (%)': hybrid_predictor.success_rate * 100 - markov_predictor.success_rate * 100
        })
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Сохраняем таблицу результатов
    results_path = f"{save_dir}/comparison_results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Создаем график сравнения
    plt.figure(figsize=(15, 8))
    
    width = 0.35
    x = np.arange(len(configs))
    
    markov_success = results_df['Markov Success Rate (%)']
    hybrid_success = results_df['Hybrid Success Rate (%)']
    
    bars1 = plt.bar(x - width/2, markov_success, width, label='Марковский предиктор', color='blue', alpha=0.7)
    bars2 = plt.bar(x + width/2, hybrid_success, width, label='Гибридный предиктор', color='green', alpha=0.7)
    
    # Добавляем значения над столбцами
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{height:.1f}%", ha='center', fontsize=9)
    
    plt.title('Сравнение успешности предикторов по конфигурациям', fontsize=14)
    plt.xlabel('Конфигурация', fontsize=12)
    plt.ylabel('Успешность (%)', fontsize=12)
    plt.xticks(x, [f"Config {i+1}" for i in range(len(configs))])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    comparison_chart_path = f"{save_dir}/comparison_chart_{timestamp}.png"
    plt.savefig(comparison_chart_path)
    plt.show()
    
    print(f"\nРезультаты сравнения сохранены в {results_path}")
    print(f"График сравнения сохранен в {comparison_chart_path}")
    
    return results_df


def main():
    # Определяем различные конфигурации для сравнения
    configs = [
        # Конфигурация 1: Базовая
        PredictorConfig(
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
        ),
        
        # Конфигурация 2: Увеличенный window_size
        PredictorConfig(
            window_size=1000,
            prediction_depth=15,
            min_confidence=0.8,
            state_length=4,
            significant_change_pct=0.4,
            use_weighted_window=False,
            quantiles=(0.1, 0.5, 0.9),
            min_samples_for_regression=10,
            confidence_threshold=0.75,
            max_coverage=0.05
        ),
        
        # Конфигурация 3: Увеличенный state_length
        PredictorConfig(
            window_size=750,
            prediction_depth=15,
            min_confidence=0.8,
            state_length=5,
            significant_change_pct=0.4,
            use_weighted_window=False,
            quantiles=(0.1, 0.5, 0.9),
            min_samples_for_regression=10,
            confidence_threshold=0.75,
            max_coverage=0.05
        ),
        
        # Конфигурация 4: Сниженный порог уверенности
        PredictorConfig(
            window_size=750,
            prediction_depth=15,
            min_confidence=0.7,
            state_length=4,
            significant_change_pct=0.4,
            use_weighted_window=False,
            quantiles=(0.1, 0.5, 0.9),
            min_samples_for_regression=10,
            confidence_threshold=0.7,
            max_coverage=0.05
        )
    ]
    
    # Определяем путь к данным
    try:
        # Пытаемся загрузить пример данных
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_data.csv')
        # Проверяем, существует ли файл
        if not os.path.exists(data_path):
            print(f"Файл примера данных не найден: {data_path}")
            print("Пожалуйста, укажите путь к файлу с данными:")
            data_path = input()
    except Exception:
        print("Пожалуйста, укажите путь к файлу с данными:")
        data_path = input()
    
    # Запускаем сравнение
    results = run_model_comparison(data_path, configs)
    
    # Выводим итоговую таблицу
    print("\nИтоговые результаты сравнения:")
    print(results)


if __name__ == "__main__":
    main()