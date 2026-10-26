import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "Не найден OPENAI_API_KEY. "
        "Создай файл .env и добавь в него ключ API."
    )


llm = ChatOpenAI(
    model="gpt-5.6",
)

parser = StrOutputParser()


# Этап 1. Анализ задачи
analysis_prompt = ChatPromptTemplate.from_template(
    """
Ты аналитик.

Проанализируй тему будущего краткого отчёта:

ТЕМА:
{topic}

Определи:
1. цель отчёта;
2. целевую аудиторию;
3. основные вопросы, которые необходимо раскрыть;
4. какой результат должен получить читатель.

Ответ дай кратко и структурированно.
"""
)

analysis_chain = analysis_prompt | llm | parser


# Этап 2. Подбор структуры
planning_prompt = ChatPromptTemplate.from_template(
    """
Ты редактор и специалист по подготовке аналитических материалов.

Исходная тема:
{topic}

Результат анализа:
{analysis}

На основании анализа создай оптимальную структуру краткого отчёта.

Укажи:
1. название;
2. основные разделы;
3. что должно быть раскрыто в каждом разделе.

Не пиши сам отчёт — только его структуру.
"""
)

planning_chain = planning_prompt | llm | parser


# Этап 3. Генерация отчёта
generation_prompt = ChatPromptTemplate.from_template(
    """
Ты профессиональный автор аналитических отчётов.

Тема:
{topic}

Анализ задачи:
{analysis}

Структура:
{plan}

Напиши краткий аналитический отчёт на русском языке.

Требования:
- следуй предложенной структуре;
- используй понятный деловой стиль;
- избегай лишней воды;
- сформулируй вывод;
- объём примерно 500–700 слов.
"""
)

generation_chain = generation_prompt | llm | parser


# Этап 4. Проверка и улучшение
review_prompt = ChatPromptTemplate.from_template(
    """
Ты редактор, выполняющий финальную проверку отчёта.

Тема:
{topic}

Черновик:
{draft}

Проверь:
- соответствует ли текст теме;
- логична ли структура;
- нет ли повторов;
- понятны ли выводы;
- выдержан ли деловой стиль.

Исправь обнаруженные недостатки.

Верни только окончательную улучшенную версию отчёта.
"""
)

review_chain = review_prompt | llm | parser


def run_chain(topic: str):
    print("\n=== ЭТАП 1. АНАЛИЗ ЗАДАЧИ ===")
    analysis = analysis_chain.invoke({
        "topic": topic
    })
    print(analysis)

    print("\n=== ЭТАП 2. ПОДБОР СТРУКТУРЫ ===")
    plan = planning_chain.invoke({
        "topic": topic,
        "analysis": analysis
    })
    print(plan)

    print("\n=== ЭТАП 3. ГЕНЕРАЦИЯ ОТЧЁТА ===")
    draft = generation_chain.invoke({
        "topic": topic,
        "analysis": analysis,
        "plan": plan
    })
    print(draft)

    print("\n=== ЭТАП 4. ПРОВЕРКА / РЕВЬЮ ===")
    final_report = review_chain.invoke({
        "topic": topic,
        "draft": draft
    })
    print(final_report)

    return final_report


if __name__ == "__main__":
    topic = input("Введите тему отчёта: ")
    run_chain(topic)