import base64
import hashlib
import streamlit as st
import sqlite3
from dotenv import load_dotenv
import os
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("📎 PDF 파일 업로드", type="pdf", label_visibility="collapsed")
    st.markdown("##### 📄 PDF 파일을 업로드하면 문서 기반 응답이 활성화됩니다.")

with col2:
    use_summary = st.toggle("요약만 보기")


# 문서 기반 응답
custom_prompt = PromptTemplate.from_template("""
    너는 한국어로 응답하는 금융 전문가야.
    문서 기반 질문에 대해 아래 기준으로 답변해줘:
                                        
    1. 질문을 충분히 이해한 후 문서 내용을 바탕으로 신뢰성 있는 정보 제공
    2. 초보자도 이해할 수 있게 설명
    3. 반드시 한국어로 응답
    4. 수치, 전략, 위험 요소 등을 구체적으로 포함

질문: {question}
---
답변:
""")


# --- 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 업로드 처리 & 벡터스토어 캐시 ---
import hashlib, threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes: bytes):
    file_hash = hashlib.md5(file_bytes).hexdigest()
    pdf_path = f"/tmp/{file_hash}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs, embeddings)
    return vs, file_hash, docs

# --- 과거 메시지 렌더 ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 미리 UI 토글
do_post = st.toggle("요약/이미지 생성 켜기", key="do_postprocess", value=False)

# --- 입력 받기 ---
if question := st.chat_input("무엇을 도와드릴까요?"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    pdf_mode, docs, retriever = False, None, None

    if uploaded_file:
        file_bytes = uploaded_file.read()  # 한 번만
        vectorstore, file_hash, docs = build_vectorstore(file_bytes)

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 6})
        base_retriever.search_type = "mmr"

        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        pdf_mode = True
        st.markdown(
            """
            <div style="background-color:#3E3B16;padding:10px;border-radius:5px;border-left:5px solid #FFD700;">
                <strong>PDF를 바탕으로 정보를 제공합니다.</strong>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#3E3B16;padding:10px;border-radius:5px;border-left:5px solid #FFD700;">
                <strong>PDF 없이 기본적인 금융 정보를 안내합니다.</strong>
            </div>
            """, unsafe_allow_html=True
        )

    # --- 답변 생성 ---
    if pdf_mode:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=False,
        )
        result = qa_chain.invoke({"query": question})
        response = result["result"]
    else:
        prompt = PromptTemplate.from_template("""
            너는 금융투자 분야에 특화된 AI야.
            아래 기준으로 한국어로만 간결하게 답해:
            1) 신뢰 가능한 출처 기반  2) 초보자 용어 풀어쓰기  3) 수치/위험요소 포함
            질문: {question}
        """)
        response = llm.predict(prompt.format(question=question))

    # --- 한 번만 출력 ---
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # --- 비동기 로그 ---
    def log_async(q, a):
        def _w():
            conn = sqlite3.connect("chat_logs.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT, answer TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            cur.execute("INSERT INTO chat_logs (question, answer) VALUES (?,?)", (q, a))
            conn.commit(); conn.close()
        threading.Thread(target=_w, daemon=True).start()
    log_async(question, response)

    # --- 요약/워드클라우드 ---
    @st.cache_data(show_spinner=False)
    def summarize_once(_docs, _hash):
        # _hash를 강제로 읽어 캐시 키로 사용
        _ = str(_hash)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(_docs)

    if do_post and pdf_mode:
        summary = summarize_once(docs, file_hash)
        st.success(summary)

        @st.cache_data(show_spinner=False)
        def generate_wordcloud_image_cached(text, _hash):
            _ = str(_hash)
            wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white").generate(text)
            buf = BytesIO(); wc.to_image().save(buf, format='PNG'); buf.seek(0); return buf
        st.image(generate_wordcloud_image_cached(summary, file_hash))

    # --- PDF 저장 버튼(중복 요약 제거, response만 PDF로) ---
    if st.button("🤖 답변을 PDF로 저장"):
        try:
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.setFont('NanumGothic', 12)

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
                if i and i % max_lines_per_page == 0:
                    c.drawText(textobject); c.showPage()
                    textobject = c.beginText(50, 750); textobject.setFont("NanumGothic", 12)
                    y = 750
                textobject.setTextOrigin(50, y); textobject.textLine(line); y -= line_height

            c.drawText(textobject); c.save()
            st.session_state.pdf_download = pdf_buffer.getvalue()
        except Exception as e:
            st.error(f"PDF 생성 중 오류 발생: {e}")

    # --- 다운로드 링크/이미지 ---
    if "pdf_download" in st.session_state:
        b64 = base64.b64encode(st.session_state["pdf_download"]).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="etf_response.pdf">📄 답변 PDF 다운로드</a>', unsafe_allow_html=True)



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

