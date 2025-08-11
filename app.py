import base64
import hashlib
import streamlit as st
import sqlite3, textwrap
from dotenv import load_dotenv
import os
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import threading
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
uploaded_files = st.file_uploader("PDF 여러 개 업로드", type="pdf", accept_multiple_files=True)
st.markdown("##### 📄 PDF 파일을 업로드하면 문서 기반 응답이 활성화됩니다.")

@st.cache_resource(show_spinner=False)
def build_vs_multi(files: tuple[bytes, ...]):
    all_docs = []
    for raw in files:
        h = hashlib.md5(raw).hexdigest()
        path = f"/tmp/{h}.pdf"
        with open(path, "wb") as f:
            f.write(raw)
        pages = PyPDFLoader(path).load()
        docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)
        all_docs.extend(docs)
    vs = FAISS.from_documents(all_docs, OpenAIEmbeddings(model="text-embedding-3-small"))
    return vs, all_docs


# 문서 기반 응답
stuff_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "너는 한국어 금융 전문가다. 아래 컨텍스트를 근거로 간결하고 정확히 답해줘.\n"
        "- 초보자도 이해 가능하게 설명\n"
        "- 수치/전략/위험요소는 구체적으로\n"
        "- 모르면 모른다고 답하기\n\n"
        "컨텍스트:\n{context}\n\n"
        "질문: {question}\n"
        "답변:"
    ),
)


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

    if uploaded_files:
        raws = tuple(f.getvalue() for f in uploaded_files)
        # 캐시 키로 쓸 결합 해시
        combined_hash = hashlib.md5(b"".join(raws)).hexdigest()

        try:
            vectorstore, docs = build_vs_multi(raws)
        except Exception as e:
            st.error(f"PDF 처리 오류: {e}")
            pdf_mode = False
        else:
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 12})
            base_retriever.search_type = "mmr"

            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
            pdf_mode = True
            st.session_state["last_file_hash"] = combined_hash  # 요약/워클 캐시용
            st.markdown(
                """
                <div style="background-color:#3E3B16;padding:10px;border-radius:5px;border-left:5px solid #FFD700;">
                    <strong>PDF를 바탕으로 정보를 제공합니다.</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="background-color:#3E3B16;padding:10px;border-radius:5px;border-left:5px solid #FFD700;">
                <strong>PDF 없이 기본적인 금융 정보를 안내합니다.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.pop("file_meta", None)
        

    # --- 답변 생성 ---
    if pdf_mode:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": stuff_prompt},
            return_source_documents=False,
        )

        try:
            result = qa_chain.invoke({"query": question})
            response = (result["result"] or "").strip()
        except Exception:
            response = ""

        
        if not response:
            try:
                rephrased = llm.predict(
                    f"다음 질문을 문서 검색에 유리하게 한국어로 한 문장으로 재표현해줘: {question}"
                ).strip()
            except Exception:
                response = question

        qa_chain_loose = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=base_retriever,
            chain_type_kwargs={"prompt": stuff_prompt},
            return_source_documents=False,
        )
        try:
            result2 = qa_chain_loose.invoke({"query": rephrased})
            response = (result2["result"] or "").strip()
        except Exception:
            response = ""

        if not response:
            try:
                summary = summarize_once(docs, st.session_state.get("last_file_hash", "nohash"))
                response = "문서에서 직접적인 매칭을 찾기 어려워 요약 기반으로 핵심을 정리했습니다:\n\n" + summary
            except Exception:
                response = "문서에서 관련 내용을 찾기 어렵습니다. 질문을 조금 더 구체화하거나 다른 PDF로 시도해 주세요."

    else:
        prompt = PromptTemplate.from_template("""
            너는 금융투자 분야에 특화된 AI야.
            아래 기준으로 한국어로만 간결하게 답해:
            1) 신뢰 가능한 출처 기반  2) 초보자 용어 풀어쓰기  3) 수치/위험요소 포함
            질문: {question}
        """)
        response = llm.predict(prompt.format(question=question))

    # 세션에 저장 + 이전 PDF 초기화
    st.session_state["last_response"] = response
    st.session_state["last_docs"] = docs
    st.session_state["last_file_hash"] = file_hash if pdf_mode else None
    st.session_state.pop("pdf_download", None) # 새로운 질문마다 이전 pdf 제거



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


# --- 항상 렌더 (PDF 생성/다운로드)---
can_export = bool(st.session_state.get("last_response"))

if st.session_state.get("do_postprocess") and st.session_state.get("last_docs"):
    @st.cache_data(show_spinner=False)
    def summarize_once(_docs, _hash):
        _ = str(_hash)
        map_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "다음 텍스트의 핵심을 한국어로 3~5개 불릿으로 요약하라.\n"
                "- 수치/위험요소는 구체적으로\n"
                "- 과도한 확정 표현 금지\n\n{text}\n\n요약:"
            ),
        )
        combine_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "아래 요약들을 한국어 단락 3~5문장으로 자연스럽게 통합하라. "
                "중복 제거하고 핵심만 남겨라.\n\n{text}\n\n최종 요약:"
            ),
        )
        chain = load_summarize_chain(
            llm, chain_type="map_reduce",
            map_prompt=map_prompt, combine_prompt=combine_prompt
        )
        return chain.run(_docs)

    summary = summarize_once(st.session_state["last_docs"], st.session_state["last_file_hash"])
    st.success(summary)

    @st.cache_data(show_spinner=False)
    def generate_wordcloud_image_cached(text:str, _hash:str):
        _ = str(_hash)
        stop = set(STOPWORDS) | {"https","http","www","com"}
        wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white", collocations=False, stopwords=stop).generate(text)
        buf = BytesIO(); wc.to_image().save(buf, format='PNG'); buf.seek(0); return buf

    hash_key = st.session_state.get("last_file_hash", "nohash")
    st.image(generate_wordcloud_image_cached(summary, hash_key), use_container_width=True)



if st.button("🤖 답변을 PDF로 저장", disabled=not can_export, use_container_width=True):
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4

        text = st.session_state.get("last_response", "")
        buf = BytesIO()

        # A4 페이지 설정
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        # 스타일 설정
        styles = getSampleStyleSheet()
        kstyle = ParagraphStyle(
            'Korean',
            parent=styles['Normal'],
            fontName='NanumGothic',
            fontSize=12,
            leading=16,
            wordWrap='CJK'
        )

        # 줄바꿈 처리
        text = text.replace("\n", "<br/>")
        story = [Paragraph(text, kstyle)]
        doc.build(story)

        st.session_state["pdf_download"] = buf.getvalue()
        st.toast("PDF 준비 완료 ✅")
    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")

pdf_bytes = st.session_state.get("pdf_download")
if pdf_bytes:
    st.download_button(
        "📄 PDF 다운로드",
        data=pdf_bytes,
        file_name="etf_response.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )