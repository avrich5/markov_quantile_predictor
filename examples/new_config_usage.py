"""
Пример использования новой системы конфигурации.
"""

import os
import sys

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from markov_quantile_predictor import (
    create_predictor, 
    create_config,
    STANDARD,
    HIGH_PRECISION,
    HIGH_VOLATILITY,
    LOW_VOLATILITY
)
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp


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
    
    # Пример 1: Создание предиктора с предустановкой
    print("\n=== Пример 1: Использование предустановки ===")
    predictor1 = create_predictor(model_type="enhanced_hybrid", preset_name="standard")
    print(f"Конфигурация предиктора: {predictor1.config}")
    
    # Пример 2: Создание предустановки с переопределением параметров
    print("\n=== Пример 2: Переопределение параметров предустановки ===")
    predictor2 = create_predictor(
        model_type="enhanced_hybrid", 
        preset_name="high_volatility",
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)  # Переопределяем квантили
    )
    print(f"Конфигурация предиктора: {predictor2.config}")
    
    # Пример 3: Создание конфигурации напрямую
    print("\n=== Пример 3: Создание конфигурации напрямую ===")
    custom_config = create_config(
        window_size=800,
        prediction_depth=10,
        quantiles=(0.1, 0.5, 0.9)
    )
    predictor3 = create_predictor(model_type="enhanced_hybrid", config=custom_config)
    print(f"Конфигурация предиктора: {predictor3.config}")
    
    # Пример 4: Использование предопределенных конфигураций из импорта
    print("\n=== Пример 4: Использование предопределенных конфигураций ===")
    # Создаем предиктор с конфигурацией для низкой волатильности
    predictor4 = create_predictor(config=create_config(preset_name="low_volatility"))
    print(f"Конфигурация для низкой волатильности: {predictor4.config}")
    
    # Запускаем предиктор на данных
    print("\n=== Запуск предиктора на данных ===")
    results = predictor1.run_on_data(prices, volumes)
    
    # Визуализируем результаты
    timestamp = get_timestamp()
    
    # Сохраняем результаты
    save_path = f"reports/enhanced_hybrid_{timestamp}.png"
    predictor1.visualize_results(prices, results, save_path)
    
    # Генерируем отчет
    report_path = f"reports/enhanced_hybrid_report_{timestamp}.md"
    predictor1.generate_report(results, report_path, prices)
    
    print(f"\nОтчет сохранен в {report_path}")
    print(f"Визуализация сохранена в {save_path}")
    print("\nАнализ завершен.")

    # Предположительное содержимое в конце файла new_config_usage.py
if __name__ == "__main__":
    main()