"""
rag_engine.py - RAG-движок для поиска информации в лекциях по имитационному моделированию.

Этот модуль использует LlamaIndex/LangChain для индексации PDF-лекций и поиска
релевантной информации по запросам пользователя.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class RAGEngine:
    """Класс для работы с RAG-движком (Retrieval-Augmented Generation)."""

    def __init__(self, knowledge_path: str = "knowledge"):
        """
        Инициализация RAG-движка.

        :param knowledge_path: путь к папке с лекциями (PDF файлы)
        """
        self.knowledge_path = Path(knowledge_path)
        self.index = None
        self.documents = []
        self._initialized = False

    def initialize(self) -> bool:
        """
        Инициализация индекса знаний из PDF-файлов.

        :return: True если успешно, False иначе
        """
        if not self.knowledge_path.exists():
            print(f"Папка с знаниями не найдена: {self.knowledge_path}")
            return False

        pdf_files = list(self.knowledge_path.glob("*.pdf"))
        
        if not pdf_files:
            print("PDF файлы не найдены в папке знаний")
            return False

        # Пока просто сохраняем список файлов для будущей интеграции
        # В полной версии здесь будет код с LlamaIndex/LangChain
        self.documents = [str(f) for f in pdf_files]
        self._initialized = True
        
        return True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантной информации по запросу.

        :param query: текстовый запрос пользователя
        :param top_k: количество результатов для возврата
        :return: список найденных фрагментов с метаинформацией
        """
        if not self._initialized:
            self.initialize()

        # Заглушка для демонстрации структуры ответа
        # В полной версии здесь будет реальный поиск через LLM
        results = []
        
        # Пример ответа для тестирования
        demo_results = [
            {
                "text": "Метод обратных функций используется для генерации случайных величин. "
                        "Если F(x) - функция распределения, то X = F^(-1)(U), где U ~ U(0,1).",
                "source": "Генерация случайных величин.pdf",
                "page": 5,
                "relevance_score": 0.95
            },
            {
                "text": "Экспоненциальное распределение моделирует время между событиями в простейшем потоке. "
                        "Плотность вероятности: f(x) = λ·e^(-λx) для x ≥ 0.",
                "source": "Моделирование непрерывных распределений.pdf",
                "page": 12,
                "relevance_score": 0.87
            },
            {
                "text": "Метод Бокса-Мюллера позволяет генерировать нормально распределенные величины. "
                        "Используются две независимые равномерные величины U1, U2.",
                "source": "Генерация случайных величин.pdf",
                "page": 18,
                "relevance_score": 0.82
            }
        ]
        
        # Возвращаем top_k результатов
        return demo_results[:top_k]

    def get_distribution_theory(self, distribution_name: str) -> Dict[str, Any]:
        """
        Получение теоретической информации о распределении из лекций.

        :param distribution_name: название распределения (на русском или английском)
        :return: словарь с информацией о распределении
        """
        # Словарь соответствия названий
        dist_mapping = {
            "exponential": ["экспоненциальное", "exponential"],
            "gamma": ["гамма", "gamma"],
            "normal": ["нормальное", "гаусса", "normal", "gauss"],
            "poisson": ["пуассона", "poisson"],
            "uniform": ["равномерное", "uniform"],
            "triangular": ["треугольное", "triangular"]
        }

        # Поиск релевантных документов
        results = self.search(f"{distribution_name} распределение формула метод генерации", top_k=5)

        return {
            "distribution": distribution_name,
            "found_in_lectures": len(results) > 0,
            "sources": [r["source"] for r in results],
            "snippets": [r["text"] for r in results],
            "note": "Для полного доступа к лекциям подключите LLM-провайдер в n8n"
        }

    def explain_method(self, method_name: str) -> str:
        """
        Объяснение метода генерации на основе лекций.

        :param method_name: название метода (например, "обратных функций", "Бокса-Мюллера")
        :return: текстовое объяснение метода
        """
        results = self.search(f"метод {method_name} генерация случайных величин", top_k=3)
        
        if results:
            explanation = "\n\n".join([r["text"] for r in results])
            return f"Информация из лекций:\n\n{explanation}"
        else:
            return "Информация о данном методе не найдена в лекциях."

    def get_available_topics(self) -> List[str]:
        """
        Получение списка доступных тем из лекций.

        :return: список тем
        """
        topics = [
            "Генерация случайных величин",
            "Метод обратных функций",
            "Метод Бокса-Мюллера",
            "Экспоненциальное распределение",
            "Гамма-распределение",
            "Нормальное распределение",
            "Распределение Пуассона",
            "Треугольное распределение",
            "Моделирование потоков событий",
            "Критерии согласия",
            "Проверка гипотез",
            "Случайные вектора"
        ]
        return topics


# Глобальный экземпляр RAG-движка
rag_engine = RAGEngine()


# Пример использования
if __name__ == "__main__":
    print("=== Тестирование RAG-движка ===\n")

    # Инициализация
    if rag_engine.initialize():
        print(f"Найдено лекций: {len(rag_engine.documents)}")
        for doc in rag_engine.documents:
            print(f"  - {os.path.basename(doc)}")
    else:
        print("Не удалось инициализировать RAG-движок")

    print("\n" + "="*50 + "\n")

    # Поиск по запросу
    query = "метод обратных функций экспоненциальное распределение"
    print(f"Поиск по запросу: '{query}'\n")
    
    results = rag_engine.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['source']}, стр. {result['page']}]")
        print(f"   Релевантность: {result['relevance_score']}")
        print(f"   Текст: {result['text'][:100]}...\n")

    print("="*50 + "\n")

    # Информация о распределении
    dist_info = rag_engine.get_distribution_theory("normal")
    print(f"Информация о распределении '{dist_info['distribution']}':")
    print(f"  Найдено в лекциях: {dist_info['found_in_lectures']}")
    print(f"  Источники: {', '.join(dist_info['sources'])}")