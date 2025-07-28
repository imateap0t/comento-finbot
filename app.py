import streamlit as st
import sqlite3
from dotenv import load_dotenv
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

# LLM 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 페이지 설정
st.set_page_config(page_title="ETF 챗봇", page_icon="💹")
st.title("💹 ETF 금융 상담 챗봇")
st.markdown("ETF에 대해 무엇이든 물어보세요!")
st.markdown("")
st.markdown(
    """
    <style>
    .chat-input textarea {
        height: 3em !important;
        border-radius: 8px;
        padding: 10px;
        font-size: 1rem;
    }
    .send-button {
        background-color: #1E90FF;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1.2em;
        border: none;
        cursor: pointer;
        margin-left: 10px;
    }
    .chat-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 업로드 및 입력
uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")

# 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if question := st.chat_input("무엇을 도와드릴까요?"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # PDF 파일 저장 여부
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        pdf_mode = True
        st.markdown(
            """
            <div style="background-color:#3E3B16;padding:10px;border-radius:5px;border-left:5px solid #FFD700;">
                <strong>PDF를 바탕으로 정보를 제공합니다.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        pdf_mode = False
        st.markdown(
            """
            <div style="background-color:#3E3B16;padding:10px;border-radius:5px;border-left:5px solid #FFD700;">
                <strong>PDF 없이 기본적인 ETF 정보를 안내합니다.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 프롬프트 생성 및 응답
    prompt = PromptTemplate.from_template("""
    너는 ETF 투자 관련 정보를 제공하는 전문가야.
    다음과 같이 세부 사항을 포함해서 제시해.
    1. 참조한 가이드라인 또는 보고서의 출처 및 페이지 정보
    2. 받은 질문과 유사한 투자자들이 관심 가질만한 질문 3가지

    질문: {question}
    답변:
    """)
    formatted_prompt = prompt.format(question=question)
    response = llm.predict(formatted_prompt)

    # 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # SQLite 연결
    conn = sqlite3.connect("chat_logs.db", check_same_thread=False)
    cursor = conn.cursor()

    # 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()

    # 답변 저장
    cursor.execute('''
    INSERT INTO chat_logs (question, answer) VALUES (?, ?)
    ''', (question, response))
    conn.commit()

    if pdf_mode:
        # 요약
        summarize_chain = load_summarize_chain(llm, chain_type="stuff")
        summary = summarize_chain.run(response)

        st.subheader("📌 요약")
        st.success(summary)

        # PDF 다운로드
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.drawString(100, 750, response[:1000])
        c.save()
        pdf_out = pdf_buffer.getvalue()

        st.download_button(
            label="답변 PDF 다운로드",
            data=pdf_out,
            file_name="etf_response.pdf",
            mime="application/pdf"
        )

        # 워드 클라우드
        st.subheader("☁️ 워드 클라우드")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(response)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
