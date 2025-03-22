"""
Скрипт для запуска анализа на реальных данных из папки data
"""

import os
import sys
import matplotlib.pyplot as plt

# Добавляем путь к родительской директории для импорта пакета
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markov_quantile_predictor import MarkovPredictor, MarkovQuantilePredictor, PredictorConfig
from markov_quantile_predictor.utils import load_data, ensure_dir, get_timestamp


def run_analysis(data_file, config=None):
    """
    Запускает анализ на указанном файле данных
    
    Параметры:
    data_file (str): путь к файлу данных
    config (PredictorConfig, optional): конфигурация предиктора
    """
    # Создаем директорию для отчетов
    ensure_dir("reports")
    
    # Настраиваем конфигурацию, если не передана
    if config is None:
        config = PredictorConfig(
            window_size=750,
            prediction_depth=15,
            min_confidence=0.8,
            state_length=4,
            significant_change_pct=0.4,
            use_weighted_window=False,
        )
    
    print(f"Анализ файла: {data_file}")
    print(f"Конфигурация: {config}")
    
    # Загружаем данные
    prices = load_data(data_file)
    
    # Создаем и запускаем марковский предиктор
    print("\nЗапуск марковского предиктора...")
    markov_predictor = MarkovPredictor(config)
    markov_results = markov_predictor.run_on_data(prices)
    
    # Создаем и запускаем гибридный предиктор
    print("\nЗапуск гибридного предиктора...")
    hybrid_predictor = MarkovQuantilePredictor(config)
    hybrid_results = hybrid_predictor.run_on_data(prices)
    
    # Визуализируем результаты
    timestamp = get_timestamp()
    
    # Получаем имя файла без пути для использования в имени отчета
    file_name = os.path.basename(data_file).replace('.csv', '')
    
    # Марковский предиктор
    print("\nРезультаты марковского предиктора:")
    print(f"- Всего предсказаний: {markov_predictor.total_predictions}")
    print(f"- Правильных предсказаний: {markov_predictor.correct_predictions}")
    print(f"- Успешность: {markov_predictor.success_rate * 100:.2f}%")
    
    markov_save_path = f"reports/{file_name}_markov_{timestamp}.png"
    markov_predictor.visualize_results(prices, markov_results, markov_save_path)
    
    # Гибридный предиктор
    print("\nРезультаты гибридного предиктора:")
    print(f"- Всего предсказаний: {hybrid_predictor.total_predictions}")
    print(f"- Правильных предсказаний: {hybrid_predictor.correct_predictions}")
    print(f"- Успешность: {hybrid_predictor.success_rate * 100:.2f}%")
    
    hybrid_save_path = f"reports/{file_name}_hybrid_{timestamp}.png"
    hybrid_predictor.visualize_results(prices, hybrid_results, hybrid_save_path)
    
    # Генерируем отчеты
    markov_report_path = f"reports/{file_name}_markov_report_{timestamp}.md"
    markov_predictor.generate_report(markov_results, markov_report_path)
    print(f"Отчет марковского предиктора сохранен в {markov_report_path}")
    
    hybrid_report_path = f"reports/{file_name}_hybrid_report_{timestamp}.md"
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
    
    plt.title(f'Сравнение успешности предикторов ({file_name})', fontsize=14)
    plt.ylabel('Успешность (%)', fontsize=12)
    plt.ylim([0, max(markov_success, hybrid_success) * 1.15])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    comparison_save_path = f"reports/{file_name}_comparison_{timestamp}.png"
    plt.savefig(comparison_save_path)
    plt.show()
    
    print(f"График сравнения сохранен в {comparison_save_path}")
    print("\nАнализ завершен.")


if __name__ == "__main__":
    # Укажите полные пути к файлам данных
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Файлы в директории data
    data_files = [
        os.path.join(data_dir, "BTC_price_100K.csv"),
        os.path.join(data_dir, "BTC_price_data.csv")
    ]
    
    # Проверяем наличие файлов
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        print("Ошибка: Не найдены файлы данных!")
        sys.exit(1)
    
    # Запускаем анализ для каждого файла
    for data_file in existing_files:
        run_analysis(data_file)
        print("\n" + "="*50 + "\n")  # Разделитель между анализами