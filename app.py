import base64
import streamlit as st
import sqlite3
from dotenv import load_dotenv
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import ttfonts

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

# LLM 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 한글 폰트 등록
font_path = os.path.join(os.path.dirname(__file__), "NanumGothic-Regular.ttf")
pdfmetrics.registerFont(TTFont('NanumGothic', font_path))

# 페이지 설정
st.set_page_config(page_title="ETF 챗봇", page_icon="💹")
st.title("💹 금융 상담 챗봇")
st.markdown("금융에 대해 무엇이든 물어보세요!")
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
                <strong>PDF 없이 기본적인 금융 정보를 안내합니다.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 프롬프트 생성 및 응답
    if pdf_mode:
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # 📌 템플릿 없이 바로 RetrievalQA 실행 (문서 기반 자동)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True  # 출처 활용 가능
        )

        result = qa_chain.invoke({"query": question})
        response = result["result"]


    else:
        # 📌 문서 없이 답할 경우, 명확하게 “신뢰 가능한 출처” 요청
        prompt = PromptTemplate.from_template("""
            너는 금융투자 분야에 특화된 AI야.
            질문자가 ETF 뿐만 아니라 예금, 적금, 주식 등도 물어볼 수 있어.
            아래 기준에 따라 응답해:
                                              
            1. 금융 기관, 신문, 보고서 등 신뢰성 높은 자료를 기반으로 작성
            2. 초보자에게 어려운 용어가 있으면 쉽게 풀이
            3. 가능하면 관련 정보가 담긴 문서나 FAQ를 요약해서 안내
            4. 출처가 있을 경우 괄호 안에 명시해 (예: (출처: 한국경제, 2025.08.03))

        질문: {question}
        ---
        답변:
        """)

        formatted_prompt = prompt.format(question=question)
        response = llm.predict(formatted_prompt)


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
        try:
            summarize_chain = load_summarize_chain(llm, chain_type="stuff")
            summary = summarize_chain.run(docs)

            st.subheader("📌 요약")
            st.success(summary)

        except Exception as e:
            st.error(f"요약 중 오류 발생: {e}")
            summary = None

        # pdf 생성
        try:
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.setFont('NanumGothic', 12)
            textobject = c.beginText(100, 750)
            textobject.setFont("NanumGothic", 12)

            max_chars_per_line = 90
            max_lines_per_page = 40

            textobject = c.beginText(50, 750)
            textobject.setFont("NanumGothic", 12)

            lines = []
            for paragraph in response.split('\n'):
                while len(paragraph) > max_chars_per_line:
                    lines.append(paragraph[:max_chars_per_line])
                    paragraph = paragraph[max_chars_per_line:]
                lines.append(paragraph)

            line_height = 16
            y = 730
            for i, line in enumerate(lines):
                if i != 0 and i % max_lines_per_page == 0:
                    c.drawText(textobject)
                    c.showPage()
                    textobject = c.beginText(50, 750)
                    textobject.setFont("NanumGothic", 12)
                    y = 750
                textobject.setTextOrigin(50, y)
                textobject.textLine(line)
                y -= line_height

            c.drawText(textobject)
            c.save()
            pdf_out = pdf_buffer.getvalue()
            st.session_state.pdf_download = pdf_out

        except Exception as e:
            st.error(f"PDF 생성 중 오류 발생: {e}")

        # 워드 클라우드 
        try: 
            @st.cache_data
            def generate_wordcloud_image(text):
                wc = WordCloud(
                    font_path=font_path,
                    width=800,
                    height=400,
                    background_color="white"
                ).generate(text)

                img_buf = BytesIO()
                wc.to_image().save(img_buf, format='PNG')
                img_buf.seek(0)
                return img_buf

            st.session_state.wordcloud_image = generate_wordcloud_image(response)

        except Exception as e:
            st.error(f"워드 클라우드 생성 중 오류 발생: {e}")

    # 응답 표시, PDF 다운로드, 워드 클라우드 출력
    def get_pdf_download_link(pdf_data):
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="etf_response.pdf">📄 답변 PDF 다운로드</a>'
        return href

    def get_wc_download_link(img_buf):
        b64 = base64.b64encode(img_buf.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="wordcloud.png">☁️ 워드클라우드 다운로드</a>'
        return href

    # 응답 표시 및 다운로드 링크 출력
    with st.chat_message("assistant"):
        st.markdown(response)

        if "pdf_download" in st.session_state:
            st.markdown(get_pdf_download_link(st.session_state["pdf_download"]), unsafe_allow_html=True)

        if "wordcloud_image" in st.session_state:
            st.image(st.session_state.wordcloud_image)
            st.markdown(get_wc_download_link(st.session_state.wordcloud_image), unsafe_allow_html=True)



# 좌측 fAq 
with st.sidebar:
    st.markdown("## 📜 투자 FAQ & 가이드")

    with st.expander("📌 자주 묻는 질문"):
        st.markdown("""
        **1. ETF란?**  
        상장지수펀드로, 지수를 추종하는 펀드입니다. 주식처럼 거래됩니다.

        **2. ETF와 펀드 차이?**  
        펀드는 하루 1번 거래, ETF는 실시간 거래가 가능합니다.

        **3. 소액으로도 가능한가요?**  
        네. ETF는 1주 단위로 매수할 수 있어, 수천 원으로도 시작할 수 있습니다.

        **4. 수익률은 어떻게 계산하나요?**  
        (현재가 - 매수가) / 매수가 × 100
        """)

    with st.expander("💡 초보자를 위한 투자 개념"):
        st.markdown("""
        **📍 분산 투자**  
        여러 자산에 나누어 투자함으로써 리스크를 줄이는 전략입니다.

        **📍 리스크란?**  
        수익이 들쭉날쭉하거나 손실이 날 가능성을 의미합니다.

        **📍 ETF 장점**  
        - 분산 투자 효과
        - 낮은 수수료
        - 다양한 자산군(주식, 채권, 금 등)

        **📍 장기 투자란?**  
        단기 시세 변동에 흔들리지 않고 일정 기간 이상 보유하여 수익을 기대하는 전략입니다.
        """)

