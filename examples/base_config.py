"""
base_config.py

Базовая конфигурация для гибридного предиктора, достигшая успешности 57.81%.
Это фиксированная отправная точка для дальнейших улучшений.
"""

import os
import sys
import pathlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

# Корректное добавление путей к модулям
file_path = pathlib.Path(__file__).resolve()
project_dir = file_path.parent.parent  # Подразумевается, что файл находится в папке examples

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Импортируем необходимые модули
from markov_quantile_predictor.models.hybrid_predictor import EnhancedHybridPredictor as HybridPredictor
from markov_quantile_predictor.predictor_config import PredictorConfig

# Выведем информацию о доступных параметрах PredictorConfig
print("Параметры конструктора PredictorConfig:")
sig = inspect.signature(PredictorConfig.__init__)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"- {param_name}: {param.default if param.default is not param.empty else 'Обязательный'}")


def ensure_dir(directory):
    """Создает директорию, если она не существует"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_timestamp():
    """Возвращает текущую метку времени в формате YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_data(data_file=None, default_size=3000, verbose=True):
    """
    Загружает данные из CSV файла или генерирует тестовые данные
    
    Параметры:
    data_file (str): путь к файлу с данными
    default_size (int): размер генерируемых данных, если файл не найден
    verbose (bool): выводить подробную информацию
    
    Возвращает:
    tuple: (prices, volumes) - массивы цен и объемов
    """
    try:
        # Пытаемся загрузить данные из указанного файла
        if data_file and os.path.exists(data_file):
            df = pd.read_csv(data_file)
        else:
            # Пытаемся найти файл с данными BTC в стандартных местах
            data_paths = [
                "data/BTC_price_data.csv",
                "data/train/btc_price_data.csv",
                "data/validation/btc_price_data.csv",
                "../data/BTC_price_data.csv",
                os.path.join(project_dir, "data/BTC_price_data.csv")
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    if verbose:
                        print(f"Найден файл данных: {path}")
                    df = pd.read_csv(path)
                    break
            else:
                # Если не нашли ни одного файла
                if verbose:
                    print("Файл с данными не найден, генерируем синтетические данные")
                np.random.seed(42)
                prices = np.cumsum(np.random.normal(0, 1, default_size)) + 1000
                return prices, None
        
        # Выделяем цены и объемы (если доступны)
        price_columns = ['price', 'close']
        price_column = next((col for col in price_columns if col in df.columns), None)
        
        if not price_column:
            price_column = df.columns[0]
            if verbose:
                print(f"Используем первую колонку: {price_column} в качестве цены")
        
        prices = df[price_column].values
        
        # Проверяем наличие данных по объемам
        volume_columns = ['volume', 'volume_base']
        volume_column = next((col for col in volume_columns if col in df.columns), None)
        
        volumes = None
        if volume_column:
            volumes = df[volume_column].values
            if verbose:
                print(f"Используем колонку {volume_column} для объемов")
        
        # Ограничиваем данные
        if len(prices) > default_size:
            prices = prices[:default_size]
            if volumes is not None:
                volumes = volumes[:default_size]
        
        if verbose:
            print(f"Загружено и подготовлено {len(prices)} точек данных")
        
        return prices, volumes
        
    except Exception as e:
        if verbose:
            print(f"Ошибка при загрузке данных: {e}")
            print("Генерируем синтетические данные...")
        
        # Генерируем тестовые данные
        np.random.seed(42)
        prices = np.cumsum(np.random.normal(0, 1, default_size)) + 1000
        return prices, None


def create_baseline_57_81_config():
    """
    Создает базовую конфигурацию, которая достигла успешности 57.81% в предыдущих тестах
    
    Возвращает:
    PredictorConfig: объект конфигурации
    """
    # Используем только наиболее важные параметры
    return PredictorConfig(
        # Основные параметры
        window_size=750,
        prediction_depth=15,
        
        # Параметры состояний
        state_length=4,
        significant_change_pct=0.004,  # 0.4%
        
        # Параметры квантильной регрессии
        quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        min_samples_for_regression=10,
        
        # Параметры предсказаний
        min_confidence=0.6,
        confidence_threshold=0.5,
        max_coverage=0.05
    )


def verify_baseline_performance(data_file=None, max_data_points=9000, verbose=True):
    """
    Проверяет производительность базовой конфигурации на указанных данных
    
    Параметры:
    data_file (str): путь к файлу с данными
    max_data_points (int): максимальное количество точек для анализа
    verbose (bool): выводить подробную информацию
    
    Возвращает:
    tuple: (предиктор, результаты, успешность)
    """
    # Создаем директорию для отчетов
    ensure_dir("reports")
    
    # Загружаем данные
    prices, volumes = load_data(
        data_file=data_file,
        default_size=max_data_points,
        verbose=verbose
    )
    
    # Создаем предиктор с базовой конфигурацией
    if verbose:
        print("\n=== Проверка базовой конфигурации (целевая успешность 57.81%) ===")
    
    baseline_config = create_baseline_57_81_config()
    predictor = HybridPredictor(baseline_config)
    
    if verbose:
        print(f"Конфигурация предиктора: {predictor.config}")
    
    # Запускаем базовую конфигурацию
    results = predictor.run_on_data(prices, volumes, verbose=verbose)
    
    # Выводим статистику
    if verbose:
        print(f"\nРезультаты базовой конфигурации:")
        print(f"- Всего предсказаний: {predictor.total_predictions}")
        print(f"- Правильных предсказаний: {predictor.correct_predictions}")
        print(f"- Успешность: {predictor.success_rate * 100:.2f}%")
        print(f"- Покрытие: {(predictor.total_predictions / len(prices)) * 100:.2f}%")
    
    # Сохраняем отчет и визуализацию
    timestamp = get_timestamp()
    try:
        baseline_save_path = f"reports/baseline_{timestamp}.png"
        predictor.visualize_results(prices, results, baseline_save_path)
        
        # Создаем отчет - безопасный способ вызова
        baseline_report_path = f"reports/baseline_report_{timestamp}.md"
        
        try:
            # Сначала пробуем с тремя аргументами
            predictor.generate_report(results, baseline_report_path, prices)
        except (TypeError, ValueError):
            try:
                # Если не получилось, пробуем с двумя аргументами
                predictor.generate_report(results, baseline_report_path)
            except Exception as e:
                if verbose:
                    print(f"Ошибка при генерации отчета: {e}")
        
        if verbose:
            print(f"\nРезультаты сохранены:")
            print(f"- Отчет: {baseline_report_path}")
            print(f"- Визуализация: {baseline_save_path}")
    except Exception as e:
        if verbose:
            print(f"Ошибка при сохранении результатов: {e}")
            import traceback
            traceback.print_exc()
    
    return predictor, results, predictor.success_rate * 100


def compare_with_custom_config(custom_config, data_file=None, max_data_points=9000, verbose=True):
    """
    Сравнивает производительность пользовательской конфигурации с базовой
    
    Параметры:
    custom_config: пользовательская конфигурация предиктора
    data_file (str): путь к файлу с данными
    max_data_points (int): максимальное количество точек для анализа
    verbose (bool): выводить подробную информацию
    
    Возвращает:
    tuple: (baseline_predictor, custom_predictor, baseline_results, custom_results)
    """
    # Загружаем данные
    prices, volumes = load_data(
        data_file=data_file,
        default_size=max_data_points,
        verbose=verbose
    )
    
    # Создаем предикторы
    baseline_config = create_baseline_57_81_config()
    baseline_predictor = HybridPredictor(baseline_config)
    custom_predictor = HybridPredictor(custom_config)
    
    if verbose:
        print("\n=== Сравнение базовой и пользовательской конфигураций ===")
        print("Базовая конфигурация:")
        print(f"- window_size: {baseline_predictor.config.window_size}")
        print(f"- prediction_depth: {baseline_predictor.config.prediction_depth}")
        print(f"- significant_change_pct: {baseline_predictor.config.significant_change_pct * 100:.2f}%")
        print(f"- confidence_threshold: {baseline_predictor.config.confidence_threshold}")
        print(f"- max_coverage: {baseline_predictor.config.max_coverage}")
        
        print("\nПользовательская конфигурация:")
        print(f"- window_size: {custom_predictor.config.window_size}")
        print(f"- prediction_depth: {custom_predictor.config.prediction_depth}")
        print(f"- significant_change_pct: {custom_predictor.config.significant_change_pct * 100:.2f}%")
        print(f"- confidence_threshold: {custom_predictor.config.confidence_threshold}")
        print(f"- max_coverage: {custom_predictor.config.max_coverage}")
    
    # Запускаем предикторы
    if verbose:
        print("\n=== Запуск предикторов ===")
    
    if verbose:
        print("\nЗапуск базовой конфигурации...")
    baseline_results = baseline_predictor.run_on_data(prices, volumes, verbose=verbose)
    
    if verbose:
        print("\nЗапуск пользовательской конфигурации...")
    custom_results = custom_predictor.run_on_data(prices, volumes, verbose=verbose)
    
    # Выводим статистику для сравнения
    if verbose:
        print("\n=== Результаты сравнения ===")
        print(f"Базовая конфигурация:")
        print(f"- Всего предсказаний: {baseline_predictor.total_predictions}")
        print(f"- Правильных предсказаний: {baseline_predictor.correct_predictions}")
        print(f"- Успешность: {baseline_predictor.success_rate * 100:.2f}%")
        print(f"- Покрытие: {(baseline_predictor.total_predictions / len(prices)) * 100:.2f}%")
        
        print(f"\nПользовательская конфигурация:")
        print(f"- Всего предсказаний: {custom_predictor.total_predictions}")
        print(f"- Правильных предсказаний: {custom_predictor.correct_predictions}")
        print(f"- Успешность: {custom_predictor.success_rate * 100:.2f}%")
        print(f"- Покрытие: {(custom_predictor.total_predictions / len(prices)) * 100:.2f}%")
        
        # Определяем улучшение или ухудшение
        success_diff = custom_predictor.success_rate - baseline_predictor.success_rate
        coverage_diff = (custom_predictor.total_predictions / len(prices)) - (baseline_predictor.total_predictions / len(prices))
        
        print("\n=== Анализ результатов ===")
        if success_diff > 0:
            print(f"📈 Успешность: УЛУЧШЕНИЕ на {success_diff * 100:.2f}%")
        elif success_diff < 0:
            print(f"📉 Успешность: УХУДШЕНИЕ на {abs(success_diff) * 100:.2f}%")
        else:
            print("🟰 Успешность: БЕЗ ИЗМЕНЕНИЙ")
        
        if coverage_diff > 0:
            print(f"📈 Покрытие: УЛУЧШЕНИЕ на {coverage_diff * 100:.2f}%")
        elif coverage_diff < 0:
            print(f"📉 Покрытие: УХУДШЕНИЕ на {abs(coverage_diff) * 100:.2f}%")
        else:
            print("🟰 Покрытие: БЕЗ ИЗМЕНЕНИЙ")
    
    # Сохраняем отчеты и визуализации
    timestamp = get_timestamp()
    try:
        # Сохраняем результаты базовой конфигурации
        baseline_save_path = f"reports/baseline_{timestamp}.png"
        baseline_predictor.visualize_results(prices, baseline_results, baseline_save_path)
        
        # Генерируем отчет
        baseline_report_path = f"reports/baseline_report_{timestamp}.md"
        try:
            baseline_predictor.generate_report(baseline_results, baseline_report_path, prices)
        except (TypeError, ValueError):
            try:
                baseline_predictor.generate_report(baseline_results, baseline_report_path)
            except Exception as e:
                if verbose:
                    print(f"Ошибка при генерации отчета для базовой конфигурации: {e}")
        
        # Сохраняем результаты пользовательской конфигурации
        custom_save_path = f"reports/custom_{timestamp}.png"
        custom_predictor.visualize_results(prices, custom_results, custom_save_path)
        
        # Генерируем отчет
        custom_report_path = f"reports/custom_report_{timestamp}.md"
        try:
            custom_predictor.generate_report(custom_results, custom_report_path, prices)
        except (TypeError, ValueError):
            try:
                custom_predictor.generate_report(custom_results, custom_report_path)
            except Exception as e:
                if verbose:
                    print(f"Ошибка при генерации отчета для пользовательской конфигурации: {e}")
        
        if verbose:
            print(f"\nРезультаты сохранены:")
            print(f"- Отчеты: {baseline_report_path}, {custom_report_path}")
            print(f"- Визуализации: {baseline_save_path}, {custom_save_path}")
    except Exception as e:
        if verbose:
            print(f"Ошибка при сохранении результатов: {e}")
            import traceback
            traceback.print_exc()
    
    return baseline_predictor, custom_predictor, baseline_results, custom_results


def main():
    """
    Демонстрирует использование базовой конфигурации
    """
    # Проверяем базовую конфигурацию
    print("\n=== ПРОВЕРКА БАЗОВОЙ КОНФИГУРАЦИИ (ЦЕЛЕВАЯ УСПЕШНОСТЬ 57.81%) ===")
    predictor, results, success_rate = verify_baseline_performance()
    
    # Проверяем, достигнута ли целевая успешность
    target_success_rate = 57.81
    diff = abs(success_rate - target_success_rate)
    
    print("\n=== РЕЗУЛЬТАТ ПРОВЕРКИ БАЗОВОЙ КОНФИГУРАЦИИ ===")
    print(f"- Целевая успешность: {target_success_rate}%")
    print(f"- Фактическая успешность: {success_rate:.2f}%")
    print(f"- Отклонение: {diff:.2f}%")
    
    if diff < 1.0:  # Допускаем небольшое отклонение в 1%
        print("\n✅ БАЗОВАЯ КОНФИГУРАЦИЯ УСПЕШНО ПОДТВЕРЖДЕНА!")
        print("   Достигнута ожидаемая успешность с допустимым отклонением.")
    else:
        print("\n❌ БАЗОВАЯ КОНФИГУРАЦИЯ НЕ ПОДТВЕРЖДЕНА")
        print("   Фактическая успешность отличается от ожидаемой более чем на 1%.")
        
        # Даем рекомендации по настройке
        if success_rate < target_success_rate:
            print("\nРекомендации для повышения успешности:")
            print("1. Уменьшите confidence_threshold для большей уверенности в предсказаниях")
            print("2. Уменьшите significant_change_pct для более чувствительного реагирования на изменения")
            print("3. Увеличьте window_size для анализа большего объема исторических данных")
        else:
            print("\nПримечание: текущая успешность превышает ожидаемую.")
            print("Это может быть результатом улучшений в алгоритме или особенностей данных.")
    
    print("\nАнализ базовой конфигурации завершен.")


if __name__ == "__main__":
    main()