"""
Валидация модели EnhancedHybridPredictor на различных наборах данных.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markov_quantile_predictor import PredictorConfig
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp
from markov_quantile_predictor.hybrid_predictor import EnhancedHybridPredictor

# Создаем директорию для результатов валидации
validation_results_dir = "validation_results_v3"
os.makedirs(validation_results_dir, exist_ok=True)


def run_validation_on_dataset(data_file, config, max_coverage, use_volumes=False):
    """
    Запускает валидацию модели на одном наборе данных
    
    Параметры:
    data_file (str): путь к файлу с данными
    config (PredictorConfig): конфигурация предиктора
    max_coverage (float): максимальное покрытие (доля предсказаний)
    use_volumes (bool): использовать ли данные объемов, если доступны
    
    Возвращает:
    dict: результаты валидации
    """
    print(f"Запуск валидации на наборе данных: {data_file}")
    
    try:
        # Загружаем данные из CSV
        print(f"Чтение файла {data_file}...")
        if not os.path.exists(data_file):
            print(f"ОШИБКА: Файл {data_file} не существует!")
            return None
            
        df = pd.read_csv(data_file)
        print(f"Загружено {len(df)} строк данных")
        
        # Выводим первые несколько строк для проверки
        print(f"Первые 3 строки данных:")
        print(df.head(3))
        print(f"Столбцы: {df.columns.tolist()}")
        
        # Определяем колонки с ценой и объемом (если есть)
        price_columns = ['price', 'close']
        price_column = next((col for col in price_columns if col in df.columns), None)
        
        if not price_column:
            print(f"ОШИБКА: Не найдена колонка с ценой в {data_file}")
            print(f"Доступные колонки: {df.columns.tolist()}")
            return None
        
        print(f"Используется колонка цены: {price_column}")
        prices = df[price_column].values
        print(f"Статистика цен: мин={min(prices)}, макс={max(prices)}, среднее={np.mean(prices)}")
        
        # Проверяем наличие данных по объемам
        volume_columns = ['volume', 'volume_base']
        volume_column = next((col for col in volume_columns if col in df.columns), None)
        
        volumes = None
        if use_volumes and volume_column:
            volumes = df[volume_column].values
            print(f"Используются данные объемов из колонки {volume_column}")
        
        # Создаем и запускаем предиктор
        print(f"Создание предиктора с конфигурацией:")
        print(f"- window_size: {config.window_size}")
        print(f"- prediction_depth: {config.prediction_depth}")
        print(f"- state_length: {config.state_length}")
        print(f"- quantiles: {config.quantiles}")
        print(f"- confidence_threshold: {config.confidence_threshold}")
        
        predictor = EnhancedHybridPredictor(config)
        print(f"Запуск предиктора на {len(prices)} точках данных...")
        
        try:
            results = predictor.run_on_data(prices, volumes)
            print(f"Выполнение завершено. Получено {len(results)} результатов.")
        except Exception as e:
            print(f"ОШИБКА при выполнении модели: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Выводим некоторые результаты для проверки
        predictions_count = sum(1 for r in results if r.get('prediction', 0) != 0)
        print(f"Количество ненулевых предсказаний: {predictions_count}")
        
        if predictions_count > 0:
            # Выводим пример предсказаний
            predictions = [r for r in results if r.get('prediction', 0) != 0]
            print(f"Пример предсказания: {predictions[0] if predictions else 'нет'}")
        
        # Вычисляем метрики
        success_rate = predictor.success_rate * 100 if predictor.total_predictions > 0 else 0
        coverage = (predictor.total_predictions / len(prices)) * 100 if len(prices) > 0 else 0
        
        print(f"Успешность: {success_rate:.2f}% ({predictor.correct_predictions}/{predictor.total_predictions})")
        print(f"Покрытие: {coverage:.2f}% ({predictor.total_predictions}/{len(prices)})")
        
        # Получаем статистику по состояниям
        state_stats = predictor.get_state_statistics()
        if not state_stats.empty:
            print(f"Наиболее частые состояния:")
            top_states = state_stats.sort_values('total', ascending=False).head(3)
            print(top_states)
        
        # Если есть квантильные предсказания, анализируем их
        quantile_stats = {}
        quantile_results = [r for r in results if 'quantile_predictions' in r]
        
        if quantile_results and len(quantile_results) > 0:
            print(f"Найдено {len(quantile_results)} записей с квантильными предсказаниями")
            
            # Выводим пример квантильных предсказаний
            if quantile_results:
                print(f"Пример квантильных предсказаний: {quantile_results[0]['quantile_predictions']}")
            
            # Процент попаданий в разные интервалы
            interval_coverage_90 = 0
            interval_coverage_50 = 0
            interval_coverage_extreme = 0
            
            for r in quantile_results:
                idx = r['index']
                if idx + predictor.config.prediction_depth >= len(prices):
                    continue
                
                # Фактическое изменение
                actual_change = prices[idx + predictor.config.prediction_depth] / prices[idx] - 1
                
                # Квантили (проверяем наличие всех нужных квантилей)
                q_preds = r['quantile_predictions']
                
                if 0.1 in q_preds and 0.9 in q_preds:
                    if q_preds[0.1] <= actual_change <= q_preds[0.9]:
                        interval_coverage_90 += 1
                
                if 0.25 in q_preds and 0.75 in q_preds:
                    if q_preds[0.25] <= actual_change <= q_preds[0.75]:
                        interval_coverage_50 += 1
                
                if 0.05 in q_preds and 0.95 in q_preds:
                    if q_preds[0.05] <= actual_change <= q_preds[0.95]:
                        interval_coverage_extreme += 1
            
            # Вычисляем проценты попаданий
            quantile_stats['interval_90_coverage'] = interval_coverage_90 / len(quantile_results) * 100 if quantile_results else 0
            quantile_stats['interval_50_coverage'] = interval_coverage_50 / len(quantile_results) * 100 if quantile_results else 0
            quantile_stats['interval_extreme_coverage'] = interval_coverage_extreme / len(quantile_results) * 100 if quantile_results else 0
            
            print(f"Покрытие интервалов:")
            print(f"- Интервал 10-90%: {quantile_stats['interval_90_coverage']:.2f}%")
            print(f"- Интервал 25-75%: {quantile_stats['interval_50_coverage']:.2f}%")
            print(f"- Интервал 5-95%: {quantile_stats['interval_extreme_coverage']:.2f}%")
            
            # Средние предсказанные изменения для разных квантилей
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
                if all(q in r['quantile_predictions'] for r in quantile_results):
                    quantile_stats[f'mean_q{int(q*100)}'] = np.mean([r['quantile_predictions'][q] * 100 for r in quantile_results])
        else:
            print("Квантильные предсказания отсутствуют")
        
        print(f"Валидация на {data_file} завершена успешно.")
        
        return {
            'dataset': os.path.basename(data_file),
            'prices': prices,
            'results': results,
            'predictor': predictor,
            'success_rate': success_rate,
            'total_predictions': predictor.total_predictions,
            'correct_predictions': predictor.correct_predictions,
            'coverage': coverage,
            'state_statistics': state_stats,
            'quantile_statistics': quantile_stats
        }
    
    except Exception as e:
        print(f"ОШИБКА при валидации на {data_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_validation_pipeline(data_dir, config=None, use_volumes=False):
    """
    Запускает валидацию на всех наборах данных
    
    Параметры:
    data_dir (str): директория с файлами данных
    config (PredictorConfig): конфигурация предиктора (если None, используется конфигурация по умолчанию)
    use_volumes (bool): использовать ли данные объемов, если доступны
    
    Возвращает:
    tuple: (результаты валидации, сводная таблица)
    """
    # Базовая конфигурация из лучшей модели
    if config is None:
        config = PredictorConfig(
            window_size=750,
            prediction_depth=15,
            min_confidence=0.6,
            state_length=4,
            significant_change_pct=0.4,
            use_weighted_window=False,
            quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),  # Уменьшенный набор квантилей
            min_samples_for_regression=10,
            confidence_threshold=0.5,
            max_coverage=0.05
        )
    
    # Находим все CSV файлы в директории с данными
    data_files = []
    try:
        print(f"Поиск CSV файлов в директории {data_dir}...")
        files_in_dir = os.listdir(data_dir)
        print(f"Всего файлов в директории: {len(files_in_dir)}")
        
        for file in files_in_dir:
            if file.endswith('.csv') and (
                file.startswith('predictor_') or 
                file.startswith('btc_') or 
                file.startswith('eth_')
            ):
                full_path = os.path.join(data_dir, file)
                if os.path.isfile(full_path):
                    file_size = os.path.getsize(full_path)
                    print(f"Найден файл: {file} ({file_size} байт)")
                    data_files.append(full_path)
    except Exception as e:
        print(f"Ошибка при чтении директории {data_dir}: {e}")

    if not data_files:
        print(f"ОШИБКА: Не найдены CSV файлы в директории {data_dir}!")
        return None, None
    
    print(f"Найдено {len(data_files)} файлов для валидации")
    
    # Ограничим количество файлов для тестирования
    # (закомментируйте эту строку, чтобы проверить все файлы)
    #data_files = data_files[:1]  # Тестируем только на первом файле
    
    # Окна для тестирования
    window_sizes = [750]  # Для ускорения тестирования используем только одно окно
    
    # Покрытие (фиксированное)
    coverage = 0.05
    
    # Для хранения результатов
    validation_results = {}
    
    # Выполняем валидацию для каждой комбинации параметров и набора данных
    timestamp = get_timestamp()
    
    for window_size in window_sizes:
        # Создаем конфигурацию с текущим размером окна
        current_config = PredictorConfig(
            window_size=window_size,
            prediction_depth=config.prediction_depth,
            min_confidence=config.min_confidence,
            state_length=config.state_length,
            significant_change_pct=config.significant_change_pct,
            use_weighted_window=config.use_weighted_window,
            quantiles=config.quantiles,
            min_samples_for_regression=config.min_samples_for_regression,
            confidence_threshold=config.confidence_threshold,
            max_coverage=coverage
        )
        
        config_name = f"ws{window_size}_cov{int(coverage*100)}"
        print(f"\n=== Валидация с параметрами: {config_name} ===")
        
        for data_file in data_files:
            dataset_name = os.path.basename(data_file).replace('predictor_', '').replace('.csv', '')
            full_name = f"{dataset_name}_{config_name}"
            print(f"\n=== Валидация: {full_name} ===")
            
            # Запускаем валидацию на текущем наборе данных
            validation_result = run_validation_on_dataset(
                data_file, current_config, coverage, use_volumes
            )
            
            if validation_result:
                validation_results[full_name] = validation_result
                
                try:
                    # Сохраняем визуализацию результатов
                    save_path = f"{validation_results_dir}/{full_name}_validation.png"
                    print(f"Сохранение визуализации в {save_path}...")
                    validation_result['predictor'].visualize_results(
                        validation_result['prices'],
                        validation_result['results'],
                        save_path
                    )
                    
                    # Создаем и сохраняем отчет
                    report_path = f"{validation_results_dir}/{full_name}_report.md"
                    print(f"Сохранение отчета в {report_path}...")
                    validation_result['predictor'].generate_report(
                        validation_result['results'],
                        report_path,
                        validation_result['prices']  # Передаем цены для анализа квантилей
                    )
                    
                    print(f"Результаты сохранены в {save_path} и {report_path}")
                except Exception as e:
                    print(f"ОШИБКА при сохранении результатов: {e}")
                    import traceback
                    traceback.print_exc()
    
    if not validation_results:
        print("Нет результатов валидации!")
        return None, None
    
    # Создаем сводную таблицу результатов
    try:
        print("\nСоздание сводной таблицы результатов...")
        summary_data = []
        for name, result in validation_results.items():
            # Базовая информация
            summary_row = {
                'Dataset': name,
                'Success Rate (%)': result['success_rate'],
                'Total Predictions': result['total_predictions'],
                'Correct Predictions': result['correct_predictions'],
                'Coverage (%)': result['coverage']
            }
            
            # Добавляем информацию о топ-состояниях
            state_stats = result['state_statistics']
            if not state_stats.empty:
                top_states = state_stats.sort_values('total', ascending=False).head(3)
                for i, (_, state_row) in enumerate(top_states.iterrows()):
                    summary_row[f'Top{i+1} State'] = state_row['state']
                    summary_row[f'Top{i+1} State Count'] = state_row['total']
                    summary_row[f'Top{i+1} State Success (%)'] = state_row['success_rate']
            
            # Добавляем информацию о квантильных прогнозах
            quantile_stats = result['quantile_statistics']
            for key, value in quantile_stats.items():
                summary_row[key] = value
            
            summary_data.append(summary_row)
        
        # Создаем DataFrame и сохраняем
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{validation_results_dir}/validation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Сводная таблица сохранена в {summary_path}")
    
        # Создаем визуализацию сравнения
        try:
            visualize_validation_results(validation_results, f"{validation_results_dir}/comparison_chart_{timestamp}.png")
        except Exception as e:
            print(f"ОШИБКА при создании визуализации сравнения: {e}")
    except Exception as e:
        print(f"ОШИБКА при создании сводной таблицы: {e}")
        import traceback
        traceback.print_exc()
        summary_df = None
    
    print(f"\n=== Валидация завершена ===")
    
    return validation_results, summary_df


def visualize_validation_results(validation_results, save_path):
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
    
    # Для квантильных интервалов (если доступны)
    interval_90_coverages = []
    interval_50_coverages = []
    
    for result in validation_results.values():
        q_stats = result['quantile_statistics']
        interval_90_coverages.append(q_stats.get('interval_90_coverage', 0))
        interval_50_coverages.append(q_stats.get('interval_50_coverage', 0))
    
    # Создаем фигуру
    plt.figure(figsize=(15, 12))
    
    # 1. График успешности по наборам данных
    plt.subplot(3, 1, 1)
    bars = plt.bar(datasets, success_rates, color='blue', alpha=0.7)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.1f}%", ha='center', fontsize=9)
    
    plt.title('Успешность предсказаний по наборам данных', fontsize=14)
    plt.xlabel('Набор данных', fontsize=12)
    plt.ylabel('Успешность (%)', fontsize=12)
    plt.ylim([0, max(success_rates) * 1.15 if success_rates and max(success_rates) > 0 else 10])
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. График покрытия по наборам данных
    plt.subplot(3, 1, 2)
    bars = plt.bar(datasets, coverages, color='green', alpha=0.7)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.1f}%", ha='center', fontsize=9)
    
    plt.title('Покрытие предсказаний по наборам данных', fontsize=14)
    plt.xlabel('Набор данных', fontsize=12)
    plt.ylabel('Покрытие (%)', fontsize=12)
    plt.ylim([0, max(coverages) * 1.15 if coverages and max(coverages) > 0 else 10])
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. График точности квантильных интервалов
    plt.subplot(3, 1, 3)
    x = np.arange(len(datasets))
    width = 0.35
    
    bar1 = plt.bar(x - width/2, interval_90_coverages, width, alpha=0.7, label='Интервал 10-90%')
    bar2 = plt.bar(x + width/2, interval_50_coverages, width, alpha=0.7, label='Интервал 25-75%')
    
    # Добавляем значения над столбцами
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{height:.1f}%", ha='center', fontsize=8)
    
    plt.title('Точность квантильных интервалов', fontsize=14)
    plt.xlabel('Набор данных', fontsize=12)
    plt.ylabel('Покрытие интервалов (%)', fontsize=12)
    plt.xticks(x, datasets, rotation=45)
    plt.legend()
    ideal_90_pct = 80  # Идеально интервал 10-90% должен покрывать 80% случаев
    ideal_50_pct = 50  # Идеально интервал 25-75% должен покрывать 50% случаев
    plt.axhline(y=ideal_90_pct, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=ideal_50_pct, color='green', linestyle='--', alpha=0.5)
    plt.ylim([0, 100])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Закрываем фигуру после сохранения


def main():
    """
    Основная функция для запуска валидации
    """
    print("Начало выполнения validate_enhanced_hybrid.py")
    
    # Попытка определить директорию с данными
    potential_paths = [
        "validation_data",  # Относительный путь
        "/Users/andriy/Visual Studio /predictor-analysis/validation_data",  # Указанный путь
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation_data"),  # Путь относительно скрипта
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_data")  # Путь в той же директории
    ]
    
    data_dir = None
    for path in potential_paths:
        print(f"Проверка пути: {path}")
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Найдена директория с данными: {path}")
            data_dir = path
            break
    
    if data_dir is None:
        print("ОШИБКА: Не удалось найти директорию с данными!")
        print("Пожалуйста, укажите путь к директории с данными:")
        user_path = input().strip()
        if os.path.exists(user_path) and os.path.isdir(user_path):
            data_dir = user_path
        else:
            print("Указанный путь не существует или не является директорией.")
            return
    
    # Конфигурация из последнего отчета с уменьшенным набором квантилей
    config = PredictorConfig(
        window_size=750,
        prediction_depth=15,
        min_confidence=0.6,
        state_length=4,
        significant_change_pct=0.4,
        use_weighted_window=False,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),  # Уменьшенный набор квантилей
        min_samples_for_regression=10,
        confidence_threshold=0.5,
        max_coverage=0.05
    )
    
    print(f"Используется конфигурация:")
    print(f"- window_size: {config.window_size}")
    print(f"- prediction_depth: {config.prediction_depth}")
    print(f"- min_confidence: {config.min_confidence}")
    print(f"- state_length: {config.state_length}")
    print(f"- significant_change_pct: {config.significant_change_pct}")
    print(f"- quantiles: {config.quantiles}")
    print(f"- confidence_threshold: {config.confidence_threshold}")
    print(f"- max_coverage: {config.max_coverage}")
    
    # Запускаем валидацию
    print("\nЗапуск пайплайна валидации...")
    validation_results, summary_df = run_validation_pipeline(data_dir, config)
    
    if summary_df is not None:
        # Выводим сводную таблицу
        print("\nСводная таблица результатов:")
        print(summary_df.to_string())
    else:
        print("\nСводная таблица отсутствует - возможно, произошла ошибка.")
    
    print("Завершение выполнения validate_enhanced_hybrid.py")


if __name__ == "__main__":
    main()