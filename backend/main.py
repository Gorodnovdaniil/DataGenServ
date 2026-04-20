"""
main.py - Оркестратор бизнес-логики backend.

Этот модуль предоставляет единый интерфейс для frontend и других компонентов системы.
Он координирует работу генератора распределений, RAG-движка и базы данных.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3
import json

from backend.generator import DistributionGenerator, get_available_distributions


class DataGenOrchestrator:
    """Основной класс оркестратора системы DataGen."""

    def __init__(self, db_path: str = "database.db"):
        """
        Инициализация оркестратора.

        :param db_path: путь к SQLite базе данных
        """
        self.db_path = db_path
        self.generator = DistributionGenerator()
        self._init_database()

    def _init_database(self):
        """Инициализация таблиц базы данных (если не существуют)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Таблица пользователей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Таблица логов генерации
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                distribution_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        conn.commit()
        conn.close()

    def generate_data(
        self,
        distribution: str,
        n: int,
        params: Dict[str, Any],
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Генерация синтетических данных указанного распределения.

        :param distribution: тип распределения (uniform, exponential, gamma, normal, poisson, triangular)
        :param n: количество величин для генерации
        :param params: параметры распределения (зависят от типа)
        :param seed: seed для воспроизводимости (опционально)
        :return: словарь с результатами генерации
        """
        # Обновляем генератор с новым seed если указан
        if seed is not None:
            self.generator = DistributionGenerator(seed=seed)
        else:
            self.generator = DistributionGenerator()

        try:
            # Выбираем метод генерации в зависимости от типа распределения
            if distribution == "uniform":
                a = params.get("a", 0.0)
                b = params.get("b", 1.0)
                data = self.generator.uniform(n, a, b)
                result_params = {"a": a, "b": b}

            elif distribution == "exponential":
                lambda_param = params.get("lambda", 1.0)
                data = self.generator.exponential(n, lambda_param)
                result_params = {"lambda": lambda_param}

            elif distribution == "gamma":
                alpha = params.get("alpha", 1.0)
                beta = params.get("beta", 1.0)
                data = self.generator.gamma(n, alpha, beta)
                result_params = {"alpha": alpha, "beta": beta}

            elif distribution == "normal":
                mu = params.get("mu", 0.0)
                sigma = params.get("sigma", 1.0)
                data = self.generator.normal(n, mu, sigma)
                result_params = {"mu": mu, "sigma": sigma}

            elif distribution == "poisson":
                lambda_param = params.get("lambda", 1.0)
                data = self.generator.poisson(n, lambda_param)
                result_params = {"lambda": lambda_param}

            elif distribution == "triangular":
                a = params.get("a", 0.0)
                b = params.get("b", 1.0)
                c = params.get("c", 0.5)
                data = self.generator.triangular(n, a, b, c)
                result_params = {"a": a, "b": b, "c": c}

            else:
                raise ValueError(f"Неизвестный тип распределения: {distribution}")

            # Вычисляем базовую статистику
            stats = self._calculate_stats(data, distribution)

            return {
                "success": True,
                "distribution": distribution,
                "parameters": result_params,
                "sample_size": n,
                "data": data[:100],  # Возвращаем первые 100 значений для UI
                "full_data_count": len(data),
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "distribution": distribution,
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_stats(self, data: List[Any], distribution: str) -> Dict[str, Any]:
        """
        Вычисление базовой статистики по сгенерированным данным.

        :param data: список сгенерированных значений
        :param distribution: тип распределения
        :return: словарь со статистикой
        """
        import numpy as np

        data_array = np.array(data)

        stats = {
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "mean": float(np.mean(data_array)),
            "std": float(np.std(data_array)),
            "median": float(np.median(data_array)),
        }

        # Для дискретных распределений добавляем уникальные значения
        if distribution == "poisson":
            unique_values = np.unique(data_array)
            stats["unique_values"] = [int(x) for x in unique_values[:20]]
            stats["value_counts"] = {
                int(k): int(v) 
                for k, v in zip(*np.unique(data_array, return_counts=True))
            }

        return stats

    def log_generation(
        self,
        user_id: Optional[int],
        distribution: str,
        params: Dict[str, Any],
        n: int
    ) -> int:
        """
        Логирование факта генерации данных в базу данных.

        :param user_id: ID пользователя (None для анонимных запросов)
        :param distribution: тип распределения
        :param params: параметры распределения
        :param n: размер выборки
        :return: ID записи в логе
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO generation_logs (user_id, distribution_type, parameters, sample_size)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, distribution, json.dumps(params), n)
        )

        log_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return log_id

    def get_distribution_info(self, distribution: str) -> Dict[str, Any]:
        """
        Получение информации о распределении (описание и допустимые параметры).

        :param distribution: тип распределения
        :return: словарь с информацией о распределении
        """
        distributions_info = {
            "uniform": {
                "name": "Равномерное распределение",
                "description": "Все значения в интервале [a, b] равновероятны",
                "parameters": {
                    "a": {"type": "float", "default": 0.0, "description": "Левая граница"},
                    "b": {"type": "float", "default": 1.0, "description": "Правая граница"}
                },
                "formula": "X ~ U(a, b)"
            },
            "exponential": {
                "name": "Экспоненциальное распределение",
                "description": "Моделирует время между событиями в пуассоновском потоке",
                "parameters": {
                    "lambda": {"type": "float", "default": 1.0, "description": "Интенсивность (λ > 0)"}
                },
                "formula": "f(x) = λ·e^(-λx)"
            },
            "gamma": {
                "name": "Гамма-распределение",
                "description": "Сумма независимых экспоненциальных величин",
                "parameters": {
                    "alpha": {"type": "float", "default": 1.0, "description": "Параметр формы (α > 0)"},
                    "beta": {"type": "float", "default": 1.0, "description": "Параметр масштаба (β > 0)"}
                },
                "formula": "f(x) = x^(α-1)·e^(-x/β) / (β^α·Γ(α))"
            },
            "normal": {
                "name": "Нормальное распределение (Гаусса)",
                "description": "Колоколообразная кривая, наиболее распространенное распределение",
                "parameters": {
                    "mu": {"type": "float", "default": 0.0, "description": "Математическое ожидание"},
                    "sigma": {"type": "float", "default": 1.0, "description": "Стандартное отклонение (σ > 0)"}
                },
                "formula": "f(x) = (1/(σ√(2π)))·e^(-(x-μ)²/(2σ²))"
            },
            "poisson": {
                "name": "Распределение Пуассона",
                "description": "Число событий за фиксированный промежуток времени",
                "parameters": {
                    "lambda": {"type": "float", "default": 1.0, "description": "Интенсивность (λ > 0)"}
                },
                "formula": "P(X=k) = (λ^k·e^(-λ)) / k!"
            },
            "triangular": {
                "name": "Треугольное распределение",
                "description": "Распределение с линейным ростом и спадом вероятности",
                "parameters": {
                    "a": {"type": "float", "default": 0.0, "description": "Минимальное значение"},
                    "b": {"type": "float", "default": 1.0, "description": "Максимальное значение"},
                    "c": {"type": "float", "default": 0.5, "description": "Мода (пик)"}
                },
                "formula": "Piecewise linear PDF"
            }
        }

        return distributions_info.get(distribution, {
            "error": f"Распределение '{distribution}' не найдено"
        })

    def get_all_distributions(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех доступных распределений с краткой информацией.

        :return: список словарей с информацией о распределениях
        """
        dist_names = get_available_distributions()
        result = []

        for name in dist_names:
            info = self.get_distribution_info(name)
            result.append({
                "id": name,
                "name": info.get("name", name),
                "description": info.get("description", "")
            })

        return result


# Глобальный экземпляр оркестратора
orchestrator = DataGenOrchestrator()


# Пример использования
if __name__ == "__main__":
    print("=== Тестирование оркестратора ===\n")

    # Получаем список распределений
    dists = orchestrator.get_all_distributions()
    print("Доступные распределения:")
    for d in dists:
        print(f"  - {d['name']} ({d['id']})")

    print("\n" + "="*50 + "\n")

    # Генерируем данные
    result = orchestrator.generate_data(
        distribution="normal",
        n=10,
        params={"mu": 5.0, "sigma": 2.0},
        seed=42
    )

    if result["success"]:
        print(f"Успешная генерация: {result['distribution']}")
        print(f"Параметры: {result['parameters']}")
        print(f"Статистика: {result['statistics']}")
        print(f"Первые 5 значений: {result['data'][:5]}")
    else:
        print(f"Ошибка: {result['error']}")