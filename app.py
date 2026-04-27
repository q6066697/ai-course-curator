"""
Streamlit UI для ИИ-куратора курса.
"""
import os
from dotenv import load_dotenv
import streamlit as st

from rag_pipeline import get_answer, create_vectorstore
from analytics import log_query

load_dotenv()


# ─── Streamlit config ────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Curator", layout="centered")
st.title("🎓 AI Course Curator")
st.caption("RAG-ассистент для студентов онлайн-курсов")


# ─── Reindex ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Управление")
    if st.button("🔄 Reindex Knowledge Base"):
        with st.spinner("Переиндексация..."):
            create_vectorstore()
        st.success("Vector DB обновлён")

    st.markdown("---")
    st.caption(
        "Куратор отвечает на вопросы по материалам курса, "
        "используя RAG-поиск по локальной базе знаний."
    )


# ─── Inputs ──────────────────────────────────────────────────────────────────
level = st.selectbox(
    "Уровень студента",
    ["beginner", "advanced"],
    help="Влияет на стиль и глубину ответа",
)
question = st.text_input(
    "Твой вопрос по курсу",
    placeholder="Например: что такое embeddings?",
)


# ─── Main flow ───────────────────────────────────────────────────────────────
if question:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY не задан. Создай .env по образцу .env.example.")
        st.stop()

    with st.spinner("Думаю..."):
        try:
            answer, context = get_answer(question, level=level)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Нажми **Reindex Knowledge Base** в сайдбаре, чтобы создать индекс.")
            st.stop()

    st.subheader("Ответ")
    st.write(answer)

    with st.expander("Контекст из базы знаний"):
        st.text(context)

    # Логирование (для дальнейшей аналитики)
    log_query(question, answer, context, level=level)
