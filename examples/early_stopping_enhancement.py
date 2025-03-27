def run_with_improved_early_stopping(self, prices, volumes=None, verbose=True, 
                                    max_time_minutes=15):
    """
    Запускает гибридную модель на всем наборе данных с ранним остановом
    
    Параметры:
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов торгов
    verbose (bool): выводить информацию о прогрессе
    max_time_minutes (int): максимальное время выполнения в минутах
    
    Возвращает:
    list: результаты предсказаний
    """
    import time
    from datetime import datetime, timedelta
    
    # Запоминаем время начала выполнения
    start_time = datetime.now()
    max_end_time = start_time + timedelta(minutes=max_time_minutes)
    
    # Предварительно вычисляем изменения цен
    self._precompute_changes(prices)
    
    results = []
    
    # Начинаем с точки, где у нас достаточно данных для анализа
    min_idx = max(self.config.window_size, self.config.state_length)
    
    # Вычисляем максимальное количество предсказаний
    max_predictions = int(len(prices) * self.config.max_coverage)
    current_predictions = 0
    
    # Для отслеживания плато
    success_rate_history = []
    best_success_rate = 0
    stable_counter = 0  # Счетчик стабильных предсказаний
    
    if verbose:
        print(f"Запуск с ранним остановом, время начала: {start_time}")
        print(f"Максимальное время выполнения: {max_time_minutes} минут")
    
    # Проходим по всем точкам
    from tqdm import tqdm
    with tqdm(total=len(prices) - min_idx - self.config.prediction_depth, 
              desc="Processing", disable=not verbose) as pbar:
        for idx in range(min_idx, len(prices) - self.config.prediction_depth):
            # Проверяем, не превышено ли максимальное время выполнения
            if datetime.now() > max_end_time:
                if verbose:
                    print(f"\nПревышено максимальное время выполнения ({max_time_minutes} минут)")
                    print(f"Остановка на {idx/len(prices)*100:.1f}% данных")
                break
            
            # Делаем предсказание
            pred_result = self.predict_at_point(prices, volumes, idx, max_predictions, current_predictions)
            prediction = pred_result['prediction']
            
            # Если предсказание не "не знаю", проверяем результат
            if prediction != 0:
                current_predictions += 1
                dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
                actual_outcome = self._determine_outcome(prices, idx, dynamic_threshold)
                
                # Пропускаем проверку, если результат незначительное изменение (0)
                if actual_outcome is None or actual_outcome == 0:
                    pbar.update(1)
                    continue
                
                is_correct = (prediction == actual_outcome)
                
                # Обновляем статистику
                self.total_predictions += 1
                if is_correct:
                    self.correct_predictions += 1
                
                # Обновляем успешность
                old_success_rate = self.success_rate
                self.success_rate = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
                
                # Сохраняем текущую успешность
                success_rate_history.append(self.success_rate)
                
                # Проверяем стабильность успешности
                if abs(old_success_rate - self.success_rate) < 0.005:  # Изменение меньше 0.5%
                    stable_counter += 1
                else:
                    stable_counter = 0
                
                # Обновляем лучшую успешность
                if self.success_rate > best_success_rate:
                    best_success_rate = self.success_rate
                
                # Сохраняем статистику для этой точки
                self.point_statistics[idx] = {
                    'correct': self.correct_predictions,
                    'total': self.total_predictions,
                    'success_rate': self.success_rate
                }
                
                # Обновляем статистику по этому состоянию
                state = pred_result['state']
                self.state_statistics[state]['total'] += 1
                if is_correct:
                    self.state_statistics[state]['correct'] += 1
                
                # Сохраняем результат с полной информацией
                result = {
                    'index': idx,
                    'price': prices[idx],
                    'prediction': prediction,
                    'actual': actual_outcome,
                    'is_correct': is_correct,
                    'confidence': pred_result['confidence'],
                    'success_rate': self.success_rate,
                    'correct_total': f"{self.correct_predictions}-{self.total_predictions}",
                    'state': pred_result['state'],
                    'state_occurrences': pred_result.get('state_occurrences', 0),
                    'quantile_predictions': pred_result.get('quantile_predictions', {})
                }
                
                # Проверяем условия для раннего останова
                if self.total_predictions >= 30:
                    # Если успешность стабилизировалась
                    if stable_counter >= 10:
                        if verbose:
                            elapsed_time = datetime.now() - start_time
                            print(f"\nУспешность стабилизировалась: {self.success_rate*100:.2f}%")
                            print(f"Время выполнения: {elapsed_time}")
                        break
            else:
                # Если предсказание "не знаю"
                result = {
                    'index': idx,
                    'price': prices[idx],
                    'prediction': 0,
                    'confidence': pred_result.get('confidence', 0.0),
                    'success_rate': self.success_rate if self.total_predictions > 0 else 0,
                    'state': pred_result.get('state'),
                    'state_occurrences': pred_result.get('state_occurrences', 0)
                }
            
            results.append(result)
            
            # Обучаем модель квантильной регрессии периодически
            if idx % 100 == 0 and idx > min_idx + self.config.prediction_depth:
                self._update_quantile_models(prices, results)
            
            # Обновляем прогресс-бар
            pbar.update(1)
            if self.total_predictions > 0:
                pbar.set_postfix({
                    'Predictions': self.total_predictions,
                    'Success Rate': f"{self.success_rate*100:.2f}%"
                })
    
    # Выводим итоговую статистику
    if verbose:
        elapsed_time = datetime.now() - start_time
        print(f"\nОбработка завершена, время выполнения: {elapsed_time}")
        print(f"Всего предсказаний: {self.total_predictions}")
        print(f"Правильных предсказаний: {self.correct_predictions}")
        print(f"Итоговая успешность: {self.success_rate*100:.2f}%")
        print(f"Общее покрытие: {(self.total_predictions / len(prices)) * 100:.4f}%")
    
    # Сохраняем ссылку на массив цен для использования в generate_report
    self.prices = prices
    
    return results