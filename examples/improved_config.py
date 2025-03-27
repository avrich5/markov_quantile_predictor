"""
improved_config.py

Расширение базовой конфигурации гибридного предиктора с целью улучшения
основных метрик (успешность и покрытие).
"""

import os
import sys
import pathlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Корректное добавление путей к модулям
file_path = pathlib.Path(__file__).resolve()
project_dir = file_path.parent.parent  # Подразумевается, что файл находится в папке examples

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Импортируем необходимые модули
from markov_quantile_predictor.models.hybrid_predictor import EnhancedHybridPredictor as HybridPredictor
from markov_quantile_predictor.predictor_config import PredictorConfig

# Импортируем функции из базового модуля
from base_config import (
    create_baseline_57_81_config, 
    compare_with_custom_config, 
    load_data, 
    ensure_dir, 
    get_timestamp
)


def create_improved_success_config():
    """
    Создает конфигурацию с целью повышения успешности предсказаний
    Основана на базовой конфигурации с целевой успешностью 57.81%
    
    Возвращает:
    PredictorConfig: объект улучшенной конфигурации
    """
    # Получаем базовую конфигурацию
    base_config = create_baseline_57_81_config()
    
    # Модифицируем параметры для повышения успешности
    return PredictorConfig(
        # Основные параметры
        window_size=base_config.window_size + 250,  # Увеличиваем размер окна для более глубокого анализа
        prediction_depth=base_config.prediction_depth,
        
        # Параметры состояний
        state_length=base_config.state_length,
        significant_change_pct=0.36,  # Уменьшаем на 10% для большей чувствительности (0.4 * 0.9 = 0.36)
        
        # Параметры квантильной регрессии
        quantiles=base_config.quantiles,
        min_samples_for_regression=base_config.min_samples_for_regression + 5,  # Требуем больше данных для обучения
        
        # Параметры предсказаний
        min_confidence=base_config.min_confidence,
        confidence_threshold=base_config.confidence_threshold + 0.1,  # Повышаем порог уверенности
        max_coverage=base_config.max_coverage * 0.8  # Уменьшаем покрытие для более точных предсказаний
    )


def create_improved_coverage_config():
    """
    Создает конфигурацию с целью повышения покрытия предсказаний
    Основана на базовой конфигурации с целевой успешностью 57.81%
    
    Возвращает:
    PredictorConfig: объект улучшенной конфигурации
    """
    # Получаем базовую конфигурацию
    base_config = create_baseline_57_81_config()
    
    # Модифицируем параметры для повышения покрытия
    return PredictorConfig(
        # Основные параметры
        window_size=base_config.window_size,
        prediction_depth=base_config.prediction_depth,
        
        # Параметры состояний
        state_length=base_config.state_length,
        significant_change_pct=0.48,  # Увеличиваем на 20% (0.4 * 1.2 = 0.48)
        
        # Параметры квантильной регрессии
        quantiles=base_config.quantiles,
        min_samples_for_regression=base_config.min_samples_for_regression - 2,  # Ослабляем требования
        
        # Параметры предсказаний
        min_confidence=base_config.min_confidence * 0.9,  # Снижаем для большего покрытия
        confidence_threshold=base_config.confidence_threshold * 0.8,  # Снижаем порог уверенности
        max_coverage=base_config.max_coverage * 2.0  # Увеличиваем максимальное покрытие
    )


def create_balanced_improvement_config():
    """
    Создает конфигурацию с целью сбалансированного улучшения успешности и покрытия
    Основана на базовой конфигурации с целевой успешностью 57.81%
    
    Возвращает:
    PredictorConfig: объект улучшенной конфигурации
    """
    # Получаем базовую конфигурацию
    base_config = create_baseline_57_81_config()
    
    # Модифицируем параметры для сбалансированного улучшения
    return PredictorConfig(
        # Основные параметры
        window_size=base_config.window_size + 100,  # Умеренно увеличиваем окно
        prediction_depth=base_config.prediction_depth,
        
        # Параметры состояний
        state_length=base_config.state_length,
        significant_change_pct=0.4,  # Оставляем как есть
        
        # Параметры квантильной регрессии
        quantiles=base_config.quantiles,
        min_samples_for_regression=base_config.min_samples_for_regression,
        
        # Параметры предсказаний
        min_confidence=base_config.min_confidence,
        confidence_threshold=base_config.confidence_threshold * 0.95,  # Немного снижаем
        max_coverage=base_config.max_coverage * 1.3  # Умеренно повышаем покрытие
    )


def create_custom_config(
    window_size=None,
    prediction_depth=None,
    state_length=None,
    significant_change_pct=None,
    quantiles=None,
    min_samples_for_regression=None,
    min_confidence=None,
    confidence_threshold=None,
    max_coverage=None
):
    """
    Создает пользовательскую конфигурацию на основе базовой,
    заменяя только указанные параметры
    
    Параметры:
    Основные параметры конфигурации, которые нужно изменить
    
    Возвращает:
    PredictorConfig: объект пользовательской конфигурации
    """
    # Получаем базовую конфигурацию
    base_config = create_baseline_57_81_config()
    
    # Создаем словарь с параметрами
    params = {
        'window_size': window_size if window_size is not None else base_config.window_size,
        'prediction_depth': prediction_depth if prediction_depth is not None else base_config.prediction_depth,
        'state_length': state_length if state_length is not None else base_config.state_length,
        'significant_change_pct': significant_change_pct if significant_change_pct is not None else base_config.significant_change_pct * 100,  # Превращаем долю в процент
        'quantiles': quantiles if quantiles is not None else base_config.quantiles,
        'min_samples_for_regression': min_samples_for_regression if min_samples_for_regression is not None else base_config.min_samples_for_regression,
        'min_confidence': min_confidence if min_confidence is not None else base_config.min_confidence,
        'confidence_threshold': confidence_threshold if confidence_threshold is not None else base_config.confidence_threshold,
        'max_coverage': max_coverage if max_coverage is not None else base_config.max_coverage
    }
    
    # Создаем конфигурацию с заданными параметрами
    return PredictorConfig(**params)


def main():
    """
    Демонстрирует использование улучшенных конфигураций
    """
    # Создаем директорию для отчетов
    ensure_dir("reports")
    
    # Создаем конфигурации
    success_config = create_improved_success_config()
    coverage_config = create_improved_coverage_config()
    balanced_config = create_balanced_improvement_config()
    
    # Выбираем конфигурацию для проверки
    print("\n=== ВЫБЕРИТЕ КОНФИГУРАЦИЮ ДЛЯ ПРОВЕРКИ ===")
    print("1. Конфигурация для повышения успешности")
    print("2. Конфигурация для повышения покрытия")
    print("3. Сбалансированная конфигурация")
    print("4. Пользовательская конфигурация")
    
    choice = input("\nВаш выбор (1-4): ")
    
    if choice == '1':
        print("\n=== ПРОВЕРКА КОНФИГУРАЦИИ ДЛЯ ПОВЫШЕНИЯ УСПЕШНОСТИ ===")
        config = success_config
    elif choice == '2':
        print("\n=== ПРОВЕРКА КОНФИГУРАЦИИ ДЛЯ ПОВЫШЕНИЯ ПОКРЫТИЯ ===")
        config = coverage_config
    elif choice == '3':
        print("\n=== ПРОВЕРКА СБАЛАНСИРОВАННОЙ КОНФИГУРАЦИИ ===")
        config = balanced_config
    elif choice == '4':
        print("\n=== СОЗДАНИЕ ПОЛЬЗОВАТЕЛЬСКОЙ КОНФИГУРАЦИИ ===")
        # Получаем параметры от пользователя
        window_size = int(input("window_size (750): ") or "750")
        prediction_depth = int(input("prediction_depth (15): ") or "15")
        significant_change_pct = float(input("significant_change_pct (0.004): ") or "0.004")
        confidence_threshold = float(input("confidence_threshold (0.005): ") or "0.005")
        max_coverage = float(input("max_coverage (0.05): ") or "0.05")
        
        config = create_custom_config(
            window_size=window_size,
            prediction_depth=prediction_depth,
            significant_change_pct=significant_change_pct,
            confidence_threshold=confidence_threshold,
            max_coverage=max_coverage
        )
    else:
        print("Неверный выбор. Используем сбалансированную конфигурацию.")
        config = balanced_config
    
    # Запускаем сравнение с базовой конфигурацией
    baseline_predictor, custom_predictor, baseline_results, custom_results = compare_with_custom_config(
        custom_config=config, 
        verbose=True
    )
    
    # Анализируем улучшения
    success_diff = custom_predictor.success_rate - baseline_predictor.success_rate
    coverage_diff = (custom_predictor.total_predictions / len(custom_results)) - (baseline_predictor.total_predictions / len(baseline_results))
    
    print("\n=== ИТОГОВЫЙ АНАЛИЗ ===")
    print(f"Базовая успешность: {baseline_predictor.success_rate * 100:.2f}%")
    print(f"Новая успешность: {custom_predictor.success_rate * 100:.2f}%")
    print(f"Изменение успешности: {success_diff * 100:+.2f}%")
    
    print(f"\nБазовое покрытие: {(baseline_predictor.total_predictions / len(baseline_results)) * 100:.2f}%")
    print(f"Новое покрытие: {(custom_predictor.total_predictions / len(custom_results)) * 100:.2f}%")
    print(f"Изменение покрытия: {coverage_diff * 100:+.2f}%")
    
    # Рассчитываем комплексную метрику эффективности (F1-подобная)
    base_f1 = 2 * baseline_predictor.success_rate * (baseline_predictor.total_predictions / len(baseline_results)) / (
        baseline_predictor.success_rate + (baseline_predictor.total_predictions / len(baseline_results))
    ) if baseline_predictor.success_rate > 0 and baseline_predictor.total_predictions > 0 else 0
    
    custom_f1 = 2 * custom_predictor.success_rate * (custom_predictor.total_predictions / len(custom_results)) / (
        custom_predictor.success_rate + (custom_predictor.total_predictions / len(custom_results))
    ) if custom_predictor.success_rate > 0 and custom_predictor.total_predictions > 0 else 0
    
    f1_diff = custom_f1 - base_f1
    
    print(f"\nБазовая комплексная метрика: {base_f1:.4f}")
    print(f"Новая комплексная метрика: {custom_f1:.4f}")
    print(f"Изменение комплексной метрики: {f1_diff:+.4f}")
    
    # Формулируем вывод
    print("\n=== ЗАКЛЮЧЕНИЕ ===")
    if f1_diff > 0:
        print("✅ УЛУЧШЕНИЕ ДОСТИГНУТО!")
        if success_diff > 0 and coverage_diff > 0:
            print("👍 Улучшены и успешность, и покрытие. Идеальный результат!")
        elif success_diff > 0:
            print("👌 Улучшена успешность при некотором снижении покрытия.")
            print("   Такой результат подходит для стратегий с высокой точностью.")
        elif coverage_diff > 0:
            print("👌 Улучшено покрытие при небольшом снижении успешности.")
            print("   Такой результат подходит для стратегий с большим количеством сделок.")
    else:
        print("❌ УЛУЧШЕНИЕ НЕ ДОСТИГНУТО")
        if success_diff <= 0 and coverage_diff <= 0:
            print("Ухудшены и успешность, и покрытие. Попробуйте другую конфигурацию.")
        elif success_diff <= 0:
            print("Ухудшена успешность, даже если увеличилось покрытие.")
            print("Попробуйте увеличить confidence_threshold и уменьшить max_coverage.")
        elif coverage_diff <= 0:
            print("Ухудшено покрытие, даже если увеличилась успешность.")
            print("Попробуйте уменьшить confidence_threshold и увеличить max_coverage.")
    
    # Сохраняем параметры успешной конфигурации
    if f1_diff > 0:
        timestamp = get_timestamp()
        config_path = f"reports/improved_config_{timestamp}.py"
        
        with open(config_path, 'w') as f:
            f.write("# Улучшенная конфигурация предиктора\n\n")
            f.write("def create_improved_config():\n")
            f.write("    \"\"\"\n")
            f.write("    Создает улучшенную конфигурацию предиктора.\n")
            f.write("    Успешность: {:.2f}%, Покрытие: {:.2f}%\n".format(
                custom_predictor.success_rate * 100,
                (custom_predictor.total_predictions / len(custom_results)) * 100
            ))
            f.write("    \"\"\"\n")
            f.write("    from markov_quantile_predictor.predictor_config import PredictorConfig\n\n")
            f.write("    return PredictorConfig(\n")
            
            # Записываем параметры конфигурации
            f.write(f"        window_size={custom_predictor.config.window_size},\n")
            f.write(f"        prediction_depth={custom_predictor.config.prediction_depth},\n")
            f.write(f"        min_confidence={custom_predictor.config.min_confidence},\n")
            f.write(f"        state_length={custom_predictor.config.state_length},\n")
            f.write(f"        significant_change_pct={custom_predictor.config.significant_change_pct * 100},\n")
            
            # Дополнительные параметры, если они доступны
            if hasattr(custom_predictor.config, 'quantiles'):
                f.write(f"        quantiles={custom_predictor.config.quantiles},\n")
            if hasattr(custom_predictor.config, 'min_samples_for_regression'):
                f.write(f"        min_samples_for_regression={custom_predictor.config.min_samples_for_regression},\n")
            f.write(f"        confidence_threshold={custom_predictor.config.confidence_threshold},\n")
            f.write(f"        max_coverage={custom_predictor.config.max_coverage}\n")
            f.write("    )\n")
        
        print(f"\nУспешная конфигурация сохранена в {config_path}")
    
    print("\nАнализ завершен.")


if __name__ == "__main__":
    main()