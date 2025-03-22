"""
Вспомогательные функции для Markov Quantile Predictor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


def load_data(csv_file):
    """
    Загружает данные из CSV файла
    
    Параметры:
    csv_file (str): путь к CSV файлу с данными
    
    Возвращает:
    numpy.array: массив цен
    """
    try:
        # Загружаем данные из CSV
        df = pd.read_csv(csv_file)
        print(f"Загружено {len(df)} строк данных")
        
        # Берем только колонку с ценой (если есть, иначе первую числовую)
        if 'price' in df.columns:
            prices = df['price'].values
        else:
            # Находим первую числовую колонку
            price_column = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
            if price_column:
                prices = df[price_column].values
            else:
                raise ValueError("Не найдена числовая колонка для цены")
                
        return prices
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("Генерирую тестовые данные...")
        
        # Генерируем тестовые данные
        np.random.seed(42)
        n_points = 10000
        prices = np.cumsum(np.random.normal(0, 1, n_points)) + 1000
        return prices


def ensure_dir(directory):
    """
    Создает директорию, если она не существует
    
    Параметры:
    directory (str): путь к директории
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_timestamp():
    """
    Возвращает текущую временную метку в формате YYYYMMDD_HHMMSS
    
    Возвращает:
    str: временная метка
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def visualize_validation_results(validation_results, save_path=None):
    """
    Создает визуализацию для сравнения результатов на разных наборах данных
    
    Параметры:
    validation_results (dict): результаты валидации
    save_path (str): путь для сохранения графика
    """
    # Подготавливаем данные для графиков
    datasets = list(validation_results.keys())
    success_rates = [result['success_rate'] for result in validation_results.values()]
    coverages = [result['coverage'] for result in validation_results.values()]
    
    # Создаем фигуру
    plt.figure(figsize=(15, 10))
    
    # 1. График успешности по наборам данных
    plt.subplot(2, 1, 1)
    bars = plt.bar(datasets, success_rates, color='blue', alpha=0.7)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.1f}%", ha='center', fontsize=9)
    
    plt.title('Успешность предсказаний по наборам данных', fontsize=14)
    plt.xlabel('Набор данных', fontsize=12)
    plt.ylabel('Успешность (%)', fontsize=12)
    plt.ylim([0, max(success_rates) * 1.15])
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. График покрытия по наборам данных
    plt.subplot(2, 1, 2)
    bars = plt.bar(datasets, coverages, color='green', alpha=0.7)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.1f}%", ha='center', fontsize=9)
    
    plt.title('Покрытие предсказаний по наборам данных', fontsize=14)
    plt.xlabel('Набор данных', fontsize=12)
    plt.ylabel('Покрытие (%)', fontsize=12)
    plt.ylim([0, max(coverages) * 1.15])
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()