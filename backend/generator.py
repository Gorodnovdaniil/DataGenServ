"""
generator.py - Модуль генерации случайных величин методами имитационного моделирования.

Все генераторы реализованы вручную на основе равномерного распределения U(0,1),
без использования готовых методов типа np.random.gamma или np.random.exponential.
Это соответствует учебным принципам имитационного моделирования.
"""

import numpy as np
from typing import List, Optional


class DistributionGenerator:
    """Класс для генерации случайных величин различных распределений."""

    def __init__(self, seed: Optional[int] = None):
        """
        Инициализация генератора.

        :param seed:_seed_ для воспроизводимости результатов (None - случайный seed)
        """
        if seed is not None:
            np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

    def uniform(self, n: int, a: float = 0.0, b: float = 1.0) -> List[float]:
        """
        Генерация равномерно распределенных величин U(a, b).
        Базовый генератор для всех остальных распределений.

        :param n: количество величин
        :param a: левая граница
        :param b: правая граница
        :return: список из n случайных величин
        """
        if a >= b:
            raise ValueError("Параметр 'a' должен быть меньше 'b'")
        u = self._rng.random(n)
        return [a + x * (b - a) for x in u]

    def exponential(self, n: int, lambda_param: float) -> List[float]:
        """
        Генерация экспоненциально распределенных величин методом обратных функций.
        
        Формула: X = -ln(U) / λ, где U ~ U(0,1)
        
        :param n: количество величин
        :param lambda_param: параметр интенсивности (λ > 0)
        :return: список из n случайных величин
        """
        if lambda_param <= 0:
            raise ValueError("Параметр λ должен быть положительным")
        
        u = self._rng.random(n)
        # Избегаем ln(0)
        u = np.maximum(u, 1e-10)
        return [-np.log(x) / lambda_param for x in u]

    def gamma(self, n: int, alpha: float, beta: float = 1.0) -> List[float]:
        """
        Генерация гамма-распределенных величин.
        
        При alpha >= 1 используется метод суммирования экспонент (для целых alpha)
        или приближенный метод для нецелых alpha.
        
        :param n: количество величин
        :param alpha: параметр формы (α > 0)
        :param beta: параметр масштаба (β > 0, по умолчанию 1)
        :return: список из n случайных величин
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Параметры α и β должны быть положительными")
        
        result = []
        
        # Если alpha целое - используем сумму экспонент
        if alpha == int(alpha):
            k = int(alpha)
            for _ in range(n):
                # Gamma(k, β) = сумма k экспоненциальных величин с λ = 1/β
                u = self._rng.random(k)
                u = np.maximum(u, 1e-10)
                value = -beta * np.sum(np.log(u))
                result.append(value)
        else:
            # Для нецелых alpha используем упрощенный метод принятия-отклонения
            # Метод основан на представлении Gamma(α) через Gamma(⌊α⌋) + остаток
            for _ in range(n):
                # Генерируем через сумму экспонент + корректировка
                k = int(alpha)
                delta = alpha - k
                
                if k > 0:
                    u = self._rng.random(k)
                    u = np.maximum(u, 1e-10)
                    value = -np.sum(np.log(u))
                else:
                    value = 0.0
                
                # Добавляем компоненту для дробной части (упрощенно)
                if delta > 0:
                    # Используем аппроксимацию для дробной части
                    u = self._rng.random()
                    u = max(u, 1e-10)
                    value += -delta * np.log(u)
                
                result.append(value * beta)
        
        return result

    def normal(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
        """
        Генерация нормально распределенных величин методом Бокса-Мюллера.
        
        :param n: количество величин
        :param mu: математическое ожидание
        :param sigma: стандартное отклонение (σ > 0)
        :return: список из n случайных величин
        """
        if sigma <= 0:
            raise ValueError("Параметр σ должен быть положительным")
        
        result = []
        # Генерируем парами
        for i in range(0, n, 2):
            u1 = self._rng.random()
            u2 = self._rng.random()
            
            # Избегаем log(0)
            u1 = max(u1, 1e-10)
            
            # Формула Бокса-Мюллера
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
            
            result.append(mu + sigma * z0)
            if i + 1 < n:
                result.append(mu + sigma * z1)
        
        return result[:n]

    def poisson(self, n: int, lambda_param: float) -> List[int]:
        """
        Генерация пуассоновски распределенных величин.
        
        Метод: генерация через произведение равномерных величин.
        
        :param n: количество величин
        :param lambda_param: параметр интенсивности (λ > 0)
        :return: список из n случайных величин (целые числа)
        """
        if lambda_param <= 0:
            raise ValueError("Параметр λ должен быть положительным")
        
        result = []
        threshold = np.exp(-lambda_param)
        
        for _ in range(n):
            k = 0
            product = 1.0
            
            while product > threshold:
                u = self._rng.random()
                u = max(u, 1e-10)
                product *= u
                k += 1
            
            result.append(k - 1)
        
        return result

    def triangular(self, n: int, a: float, b: float, c: float) -> List[float]:
        """
        Генерация треугольного распределения методом обратных функций.
        
        :param n: количество величин
        :param a: минимальное значение
        :param b: максимальное значение
        :param c: мода (a ≤ c ≤ b)
        :return: список из n случайных величин
        """
        if not (a <= c <= b):
            raise ValueError("Должно выполняться условие a ≤ c ≤ b")
        if a >= b:
            raise ValueError("Параметр 'a' должен быть меньше 'b'")
        
        result = []
        f_c = (c - a) / (b - a)  # F(c)
        
        for _ in range(n):
            u = self._rng.random()
            
            if u < f_c:
                # Левая ветвь
                value = a + np.sqrt(u * (b - a) * (c - a))
            else:
                # Правая ветвь
                value = b - np.sqrt((1 - u) * (b - a) * (b - c))
            
            result.append(value)
        
        return result

    def generate_stream(self, n: int, lambda_param: float) -> List[float]:
        """
        Генерация потока событий (времена наступления событий).
        
        Пуассоновский поток: интервалы между событиями ~ Exp(λ)
        
        :param n: количество событий в потоке
        :param lambda_param: интенсивность потока
        :return: список времен наступления событий (накопленная сумма)
        """
        intervals = self.exponential(n, lambda_param)
        
        times = []
        current_time = 0.0
        for interval in intervals:
            current_time += interval
            times.append(current_time)
        
        return times


def get_available_distributions() -> List[str]:
    """Возвращает список доступных распределений."""
    return [
        "uniform",
        "exponential", 
        "gamma",
        "normal",
        "poisson",
        "triangular"
    ]


# Пример использования для тестирования
if __name__ == "__main__":
    gen = DistributionGenerator(seed=42)
    
    print("=== Тестирование генераторов ===\n")
    
    # Равномерное
    uniform_data = gen.uniform(5, 0, 10)
    print(f"Uniform(0, 10): {uniform_data}")
    
    # Экспоненциальное
    exp_data = gen.exponential(5, lambda_param=2.0)
    print(f"Exponential(λ=2.0): {exp_data}")
    
    # Гамма
    gamma_data = gen.gamma(5, alpha=3.0, beta=2.0)
    print(f"Gamma(α=3.0, β=2.0): {gamma_data}")
    
    # Нормальное
    normal_data = gen.normal(5, mu=0.0, sigma=1.0)
    print(f"Normal(μ=0, σ=1): {normal_data}")
    
    # Пуассона
    poisson_data = gen.poisson(5, lambda_param=3.0)
    print(f"Poisson(λ=3.0): {poisson_data}")
    
    # Треугольное
    triangular_data = gen.triangular(5, a=0, b=10, c=5)
    print(f"Triangular(a=0, b=10, c=5): {triangular_data}")
    
    # Поток событий
    stream = gen.generate_stream(5, lambda_param=1.0)
    print(f"Event stream (λ=1.0): {stream}")