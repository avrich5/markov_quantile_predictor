"""
Улучшенный пример использования гибридного предиктора.
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


def create_summary_report(predictor1, predictor3, standard_results, custom_results, prices, timestamp):
    """
    Создает сводный отчет о работе предикторов с безопасной обработкой квантилей
    
    Параметры:
    predictor1: стандартный предиктор
    predictor3: пользовательский предиктор
    standard_results: результаты стандартного предиктора
    custom_results: результаты пользовательского предиктора
    prices: массив цен
    timestamp: метка времени
    
    Возвращает:
    str: путь к созданному отчету
    """
    report_path = f"reports/summary_report_{timestamp}.md"
    
    # Считаем статистику для стандартного предиктора
    standard_up_count = sum(1 for r in standard_results if r.get('prediction') == 1)
    standard_down_count = sum(1 for r in standard_results if r.get('prediction') == 2)
    standard_neutral_count = sum(1 for r in standard_results if r.get('prediction') == 0)
    
    standard_up_correct = sum(1 for r in standard_results if r.get('prediction') == 1 and r.get('is_correct', False))
    standard_down_correct = sum(1 for r in standard_results if r.get('prediction') == 2 and r.get('is_correct', False))
    
    standard_up_success_rate = standard_up_correct / standard_up_count * 100 if standard_up_count > 0 else 0
    standard_down_success_rate = standard_down_correct / standard_down_count * 100 if standard_down_count > 0 else 0
    
    # Считаем статистику для пользовательского предиктора
    custom_up_count = sum(1 for r in custom_results if r.get('prediction') == 1)
    custom_down_count = sum(1 for r in custom_results if r.get('prediction') == 2)
    custom_neutral_count = sum(1 for r in custom_results if r.get('prediction') == 0)
    
    custom_up_correct = sum(1 for r in custom_results if r.get('prediction') == 1 and r.get('is_correct', False))
    custom_down_correct = sum(1 for r in custom_results if r.get('prediction') == 2 and r.get('is_correct', False))
    
    custom_up_success_rate = custom_up_correct / custom_up_count * 100 if custom_up_count > 0 else 0
    custom_down_success_rate = custom_down_correct / custom_down_count * 100 if custom_down_count > 0 else 0
    
    # Формируем отчет
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Сводный отчет о работе предикторов\n\n")
        
        f.write("## Сравнение конфигураций\n\n")
        f.write("| Параметр | Стандартная конфигурация | Пользовательская конфигурация |\n")
        f.write("|----------|--------------------------|------------------------------|\n")
        f.write(f"| window_size | {predictor1.config.window_size} | {predictor3.config.window_size} |\n")
        f.write(f"| prediction_depth | {predictor1.config.prediction_depth} | {predictor3.config.prediction_depth} |\n")
        f.write(f"| significant_change_pct | {predictor1.config.significant_change_pct*100:.2f}% | {predictor3.config.significant_change_pct*100:.2f}% |\n")
        f.write(f"| confidence_threshold | {predictor1.config.confidence_threshold} | {predictor3.config.confidence_threshold} |\n")
        f.write(f"| max_coverage | {predictor1.config.max_coverage} | {predictor3.config.max_coverage} |\n")
        f.write(f"| quantiles | {predictor1.config.quantiles} | {predictor3.config.quantiles} |\n\n")
        
        f.write("## Сравнение результатов\n\n")
        f.write("| Метрика | Стандартная конфигурация | Пользовательская конфигурация |\n")
        f.write("|---------|--------------------------|------------------------------|\n")
        f.write(f"| Всего предсказаний | {predictor1.total_predictions} | {predictor3.total_predictions} |\n")
        f.write(f"| Правильных предсказаний | {predictor1.correct_predictions} | {predictor3.correct_predictions} |\n")
        f.write(f"| Успешность | {predictor1.success_rate*100:.2f}% | {predictor3.success_rate*100:.2f}% |\n")
        f.write(f"| Покрытие | {(predictor1.total_predictions/len(prices))*100:.2f}% | {(predictor3.total_predictions/len(prices))*100:.2f}% |\n\n")
        
        f.write("## Распределение предсказаний\n\n")
        f.write("### Стандартная конфигурация\n\n")
        f.write(f"- Рост: {standard_up_count} ({standard_up_count/len(standard_results)*100:.2f}%)\n")
        f.write(f"- Падение: {standard_down_count} ({standard_down_count/len(standard_results)*100:.2f}%)\n")
        f.write(f"- Не знаю: {standard_neutral_count} ({standard_neutral_count/len(standard_results)*100:.2f}%)\n\n")
        
        if standard_up_count > 0:
            f.write(f"- Успешность предсказаний роста: {standard_up_correct}/{standard_up_count} ({standard_up_success_rate:.2f}%)\n")
        if standard_down_count > 0:
            f.write(f"- Успешность предсказаний падения: {standard_down_correct}/{standard_down_count} ({standard_down_success_rate:.2f}%)\n\n")
        
        f.write("### Пользовательская конфигурация\n\n")
        f.write(f"- Рост: {custom_up_count} ({custom_up_count/len(custom_results)*100:.2f}%)\n")
        f.write(f"- Падение: {custom_down_count} ({custom_down_count/len(custom_results)*100:.2f}%)\n")
        f.write(f"- Не знаю: {custom_neutral_count} ({custom_neutral_count/len(custom_results)*100:.2f}%)\n\n")
        
        if custom_up_count > 0:
            f.write(f"- Успешность предсказаний роста: {custom_up_correct}/{custom_up_count} ({custom_up_success_rate:.2f}%)\n")
        if custom_down_count > 0:
            f.write(f"- Успешность предсказаний падения: {custom_down_correct}/{custom_down_count} ({custom_down_success_rate:.2f}%)\n\n")
        
        # Анализ топ-состояний
        f.write("## Топ-состояния\n\n")
        f.write("### Стандартная конфигурация\n\n")
        standard_state_stats = predictor1.get_state_statistics()
        if not standard_state_stats.empty:
            top_states = standard_state_stats.head(5).to_markdown(index=False)
            f.write(top_states + "\n\n")
        else:
            f.write("Нет данных о состояниях\n\n")
        
        f.write("### Пользовательская конфигурация\n\n")
        custom_state_stats = predictor3.get_state_statistics()
        if not custom_state_stats.empty:
            top_states = custom_state_stats.head(5).to_markdown(index=False)
            f.write(top_states + "\n\n")
        else:
            f.write("Нет данных о состояниях\n\n")
        
        # Анализ квантильных предсказаний (если есть)
        f.write("## Квантильная регрессия\n\n")
        
        # Безопасный сбор статистики квантильных предсказаний для стандартной конфигурации
        standard_quantile_results = [r for r in standard_results if 'quantile_predictions' in r and r['quantile_predictions']]
        if standard_quantile_results:
            f.write("### Стандартная конфигурация\n\n")
            f.write(f"- Количество предсказаний с квантильной регрессией: {len(standard_quantile_results)}\n")
            
            # Безопасно получаем квантили
            first_result = standard_quantile_results[0]['quantile_predictions']
            available_quantiles = sorted(first_result.keys())
            
            f.write("\n**Доступные квантили:** " + ", ".join([f"{q}" for q in available_quantiles]) + "\n\n")
            
            f.write("**Средние предсказанные изменения:**\n")
            for q in available_quantiles:
                mean_q = np.mean([r['quantile_predictions'][q] * 100 for r in standard_quantile_results])
                f.write(f"- Средний квантиль {q*100:.0f}%: {mean_q:.2f}%\n")
            
            # Ширина интервалов (только если есть нужные квантили)
            if 0.1 in available_quantiles and 0.9 in available_quantiles:
                mean_interval = np.mean([(r['quantile_predictions'][0.9] - r['quantile_predictions'][0.1]) * 100 for r in standard_quantile_results])
                f.write(f"- Средняя ширина интервала [10%, 90%]: {mean_interval:.2f}%\n\n")
        
        # Безопасный сбор статистики квантильных предсказаний для пользовательской конфигурации
        custom_quantile_results = [r for r in custom_results if 'quantile_predictions' in r and r['quantile_predictions']]
        if custom_quantile_results:
            f.write("### Пользовательская конфигурация\n\n")
            f.write(f"- Количество предсказаний с квантильной регрессией: {len(custom_quantile_results)}\n")
            
            # Безопасно получаем квантили
            first_result = custom_quantile_results[0]['quantile_predictions']
            available_quantiles = sorted(first_result.keys())
            
            f.write("\n**Доступные квантили:** " + ", ".join([f"{q}" for q in available_quantiles]) + "\n\n")
            
            f.write("**Средние предсказанные изменения:**\n")
            for q in available_quantiles:
                mean_q = np.mean([r['quantile_predictions'][q] * 100 for r in custom_quantile_results])
                f.write(f"- Средний квантиль {q*100:.0f}%: {mean_q:.2f}%\n")
            
            # Ширина интервалов (только если есть нужные квантили)
            if 0.1 in available_quantiles and 0.9 in available_quantiles:
                mean_interval = np.mean([(r['quantile_predictions'][0.9] - r['quantile_predictions'][0.1]) * 100 for r in custom_quantile_results])
                f.write(f"- Средняя ширина интервала [10%, 90%]: {mean_interval:.2f}%\n\n")
        
        f.write("## Заключение\n\n")
        
        # Определяем, какая конфигурация лучше
        if predictor3.success_rate > predictor1.success_rate and predictor3.total_predictions > 0:
            success_diff = (predictor3.success_rate - predictor1.success_rate) * 100
            f.write(f"Пользовательская конфигурация **превосходит** стандартную по успешности на {success_diff:.2f}%.\n")
        elif predictor1.success_rate > predictor3.success_rate and predictor1.total_predictions > 0:
            success_diff = (predictor1.success_rate - predictor3.success_rate) * 100
            f.write(f"Стандартная конфигурация **превосходит** пользовательскую по успешности на {success_diff:.2f}%.\n")
        else:
            f.write("Обе конфигурации показывают **одинаковую успешность** или недостаточно данных для сравнения.\n")
        
        # Сравниваем покрытие
        standard_coverage = (predictor1.total_predictions / len(prices)) * 100
        custom_coverage = (predictor3.total_predictions / len(prices)) * 100
        
        if custom_coverage > standard_coverage:
            coverage_diff = custom_coverage - standard_coverage
            f.write(f"Пользовательская конфигурация обеспечивает **большее покрытие** на {coverage_diff:.2f}%.\n")
        elif standard_coverage > custom_coverage:
            coverage_diff = standard_coverage - custom_coverage
            f.write(f"Стандартная конфигурация обеспечивает **большее покрытие** на {coverage_diff:.2f}%.\n")
        else:
            f.write("Обе конфигурации показывают **одинаковое покрытие**.\n")
    
    return report_path


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
        
        # Ограничиваем данные для ускорения анализа
        max_data_points = 3000  # Анализируем только первые 3000 точек
        prices = prices[:max_data_points]
        if volumes is not None:
            volumes = volumes[:max_data_points]
        
        print(f"Загружено и подготовлено {len(prices)} точек данных")
        
    except Exception as e:
        print(f"Не удалось загрузить данные: {e}")
        # Генерируем тестовые данные
        np.random.seed(42)
        n_points = 3000  # Меньше точек для ускорения
        prices = np.cumsum(np.random.normal(0, 1, n_points)) + 1000
        volumes = None
    
    # Пример 1: Улучшенная стандартная конфигурация
    print("\n=== Стандартная конфигурация с исправлениями ===")
    predictor1 = create_predictor(
        model_type="enhanced_hybrid", 
        preset_name="standard",
        significant_change_pct=0.01,     # 1% изменения
        confidence_threshold=0.2,        # Низкий порог уверенности
        max_coverage=0.2                 # Повышенное покрытие 
    )
    print(f"Конфигурация предиктора: {predictor1.config}")
    
    # Пример 2: Пользовательская расширенная конфигурация
    print("\n=== Расширенная пользовательская конфигурация ===")
    custom_config = create_config(
        window_size=500,                 # Меньшее окно для быстрого отклика
        prediction_depth=10,             # Меньшая глубина для своевременных предсказаний
        significant_change_pct=0.008,    # 0.8% порог изменения
        quantiles=(0.1, 0.5, 0.9),       # Стандартные квантили
        confidence_threshold=0.15,       # Очень низкий порог уверенности
        max_coverage=0.25                # Высокое покрытие
    )
    predictor3 = create_predictor(model_type="enhanced_hybrid", config=custom_config)
    print(f"Конфигурация предиктора: {predictor3.config}")
    
    # Диагностика key параметров
    print("\n=== Диагностика ключевых параметров ===")
    print("Стандартная конфигурация:")
    print(f"- significant_change_pct: {predictor1.config.significant_change_pct} (в процентах: {predictor1.config.significant_change_pct*100:.2f}%)")
    print(f"- confidence_threshold: {predictor1.config.confidence_threshold}")
    print(f"- max_coverage: {predictor1.config.max_coverage}")
    
    print("\nПользовательская конфигурация:")
    print(f"- significant_change_pct: {predictor3.config.significant_change_pct} (в процентах: {predictor3.config.significant_change_pct*100:.2f}%)")
    print(f"- confidence_threshold: {predictor3.config.confidence_threshold}")
    print(f"- max_coverage: {predictor3.config.max_coverage}")
    
    # Запускаем пользовательскую конфигурацию
    print("\n=== Запуск предиктора с пользовательской конфигурацией ===")
    custom_results = predictor3.run_on_data(prices, volumes, verbose=True)
    
    # Запускаем стандартную конфигурацию
    print("\n=== Запуск предиктора со стандартной конфигурацией ===")
    standard_results = predictor1.run_on_data(prices, volumes, verbose=True)
    
    # Выводим статистику для сравнения
    print("\n=== Сравнение результатов ===")
    print(f"Пользовательская конфигурация:")
    print(f"- Всего предсказаний: {predictor3.total_predictions}")
    print(f"- Правильных предсказаний: {predictor3.correct_predictions}")
    print(f"- Успешность: {predictor3.success_rate * 100:.2f}%")
    print(f"- Покрытие: {(predictor3.total_predictions / len(prices)) * 100:.2f}%")
    
    print(f"\nСтандартная конфигурация:")
    print(f"- Всего предсказаний: {predictor1.total_predictions}")
    print(f"- Правильных предсказаний: {predictor1.correct_predictions}")
    print(f"- Успешность: {predictor1.success_rate * 100:.2f}%")
    print(f"- Покрытие: {(predictor1.total_predictions / len(prices)) * 100:.2f}%")
    
    # Создаем визуализации и отчеты
    timestamp = get_timestamp()
    
    # Сохраняем отчеты
    try:
        # Сохраняем результаты пользовательской конфигурации
        custom_save_path = f"reports/enhanced_hybrid_custom_{timestamp}.png"
        predictor3.visualize_results(prices, custom_results, custom_save_path)

        # Генерируем отчет
        custom_report_path = f"reports/enhanced_hybrid_custom_report_{timestamp}.md"
        # predictor3.generate_report(custom_results, custom_report_path, prices)
        custom_report = predictor3.generate_report(custom_results, custom_report_path)



        # Сохраняем результаты стандартной конфигурации для сравнения
        standard_save_path = f"reports/enhanced_hybrid_standard_{timestamp}.png"
        predictor1.visualize_results(prices, standard_results, standard_save_path)

        # Генерируем отчет
        standard_report_path = f"reports/enhanced_hybrid_standard_report_{timestamp}.md"
        predictor1.generate_report(standard_results, standard_report_path, prices)
        
        # Создаем сводный отчет
        summary_report_path = create_summary_report(
            predictor1, predictor3, 
            standard_results, custom_results, 
            prices, timestamp
        )

        print(f"\nОтчеты сохранены в:")
        print(f"- {custom_report_path} (пользовательская конфигурация)")
        print(f"- {standard_report_path} (стандартная конфигурация)")
        print(f"- {summary_report_path} (сводный отчет)")
        print(f"\nВизуализация сохранена в:")
        print(f"- {custom_save_path} (пользовательская конфигурация)")
        print(f"- {standard_save_path} (стандартная конфигурация)")
    except Exception as e:
        print(f"Ошибка при создании отчетов: {e}")
        import traceback
        traceback.print_exc()
    
        print("\nАнализ завершен.")

if __name__ == "__main__":
    main()