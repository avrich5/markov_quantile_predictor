"""
Модуль с реализацией квантильной регрессии.
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


class QuantileRegressionModel:
    """
    Модель для предсказания квантилей будущего изменения цены.
    """
    
    def __init__(self, quantiles=(0.1, 0.5, 0.9), alpha=0.1):
        """
        Инициализация модели квантильной регрессии
        
        Параметры:
        quantiles (tuple): квантили для предсказания
        alpha (float): параметр регуляризации для модели
        """
        self.quantiles = quantiles
        self.alpha = alpha
        self.models = {}  # Словарь с моделями для разных квантилей
        self.scaler = StandardScaler()  # Нормализация признаков
        self.is_fitted = False  # Флаг, указывающий, обучена ли модель
    
    def fit(self, X, y):
        """
        Обучает модель на исторических данных
        
        Параметры:
        X (numpy.array): признаки (каждая строка - вектор признаков для одного наблюдения)
        y (numpy.array): целевые значения (процентное изменение цены)
        
        Возвращает:
        self: обученная модель
        """
        # Проверяем, что данных достаточно для обучения
        if len(X) < 5:
            self.is_fitted = False
            return self
        
        # Нормализуем признаки
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем модель для каждого квантиля
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, alpha=self.alpha, solver='highs')
            model.fit(X_scaled, y)
            self.models[q] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Делает предсказание для новых данных
        
        Параметры:
        X (numpy.array): признаки для предсказания
        
        Возвращает:
        dict: предсказания для разных квантилей
        """
        if not self.is_fitted:
            return None
        
        # Нормализуем признаки
        X_scaled = self.scaler.transform(X)
        
        # Делаем предсказания для каждого квантиля
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)
        
        return predictions
    
    def predict_single(self, X):
        """
        Делает предсказание для одного наблюдения
        
        Параметры:
        X (numpy.array): вектор признаков
        
        Возвращает:
        dict: предсказания для разных квантилей
        """
        if not self.is_fitted:
            return None
        
        # Преобразуем вектор признаков в 2D массив
        X_reshaped = X.reshape(1, -1) if len(X.shape) == 1 else X
        
        # Нормализуем признаки
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Делаем предсказания для каждого квантиля
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)[0]
        
        return predictions