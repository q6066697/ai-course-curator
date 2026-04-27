"""
RAG-пайплайн ИИ-куратора.

Использует LangChain + FAISS + OpenAI:
- Загружает .txt-файлы из data/course_docs/
- Делит на чанки и эмбедит через text-embedding-ada-002
- Сохраняет в локальный FAISS-индекс
- При запросе достаёт top-3 релевантных фрагмента и передаёт в LLM
"""
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from prompts import build_prompt

load_dotenv()

DATA_PATH = "data/course_docs"
DB_PATH = "vectorstore"
TOP_K = 3
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def get_embeddings() -> OpenAIEmbeddings:
    """Создаёт OpenAI-эмбеддер. Ключ берётся из окружения автоматически."""
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def load_documents() -> list:
    """Читает все .txt-файлы из DATA_PATH."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Папка с курсом не найдена: {DATA_PATH}. "
            f"Создай её и положи туда .txt-файлы лекций."
        )

    docs = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(
                os.path.join(DATA_PATH, filename),
                encoding="utf-8",
            )
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"В {DATA_PATH} нет .txt-файлов.")
    return docs


def create_vectorstore() -> None:
    """Полностью пересобирает FAISS-индекс из текущих документов."""
    docs = load_documents()
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print(f"Vectorstore создан: {len(chunks)} чанков сохранено в {DB_PATH}")


def load_vectorstore() -> FAISS:
    """Загружает существующий FAISS-индекс с диска."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"FAISS-индекс не найден в {DB_PATH}. "
            f"Запусти create_vectorstore() или нажми кнопку 'Reindex' в UI."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_context(query: str, k: int = TOP_K) -> str:
    """Возвращает top-k релевантных чанков, склеенных в один текст."""
    db = load_vectorstore()
    docs = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def get_answer(question: str, level: str = "beginner") -> tuple[str, str]:
    """
    Главная функция. Возвращает (ответ, контекст) для логирования и UI.

    Args:
        question: вопрос студента
        level: 'beginner' или 'advanced' — пробрасывается в промпт

    Returns:
        (answer_text, retrieved_context)
    """
    context = retrieve_context(question)
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = build_prompt(question=question, context=context, level=level)
    response = llm.invoke(prompt)
    return response.content, context


if __name__ == "__main__":
    print("Создаю FAISS-индекс из data/course_docs/...")
    create_vectorstore()
