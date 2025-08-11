import hashlib
import os
import sqlite3
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

# ====== LLM & LangChain (LangChain 0.3.27) ======
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# ====== PDF/시각화 ======
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from wordcloud import WordCloud, STOPWORDS

# ====== 기본 설정 ======
load_dotenv()

st.set_page_config(page_title="ETF 챗봇", page_icon="💹")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.stop()  # API 키 없으면 중단

# LLM 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# UI 헤더
st.title("💹 금융 상담 챗봇")
st.markdown("금융에 대해 무엇이든 물어보세요!")

# 한글 폰트 등록
font_path = os.path.join(os.path.dirname(__file__), "NanumGothic-Regular.ttf")
pdfmetrics.registerFont(TTFont("NanumGothic", font_path))

# 간단 CSS
st.markdown(
    """
    <style>
    .chat-input textarea { height: 3em !important; border-radius: 8px; padding: 10px; font-size: 1rem; }
    .send-button { background-color: #1E90FF; color: white; border-radius: 6px; padding: 0.5em 1.2em; border: none; cursor: pointer; margin-left: 10px; }
    .chat-row { display: flex; align-items: center; gap: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== 세션 ======
if "messages" not in st.session_state:
    st.session_state.messages = []

# ====== 업로드 위젯 ======
uploaded_files = st.file_uploader("PDF 여러 개 업로드", type="pdf", accept_multiple_files=True)
st.markdown("##### 📄 PDF 파일을 업로드하면 문서 기반 응답이 활성화됩니다.")

# ====== 벡터스토어 빌더 (멀티 PDF, 견고화) ======
@st.cache_resource(show_spinner=False)
def build_vs_multi(
    files: tuple[bytes, ...],
    file_names: tuple[str, ...],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model: str = "text-embedding-3-small",
):
    # 캐시 키 안정화: 파일 해시 + 파라미터
    _ = (chunk_size, chunk_overlap, embed_model, tuple(hashlib.md5(b).hexdigest() for b in files))

    all_docs = []
    tmp_paths = []
    try:
        for raw, fname in zip(files, file_names):
            if not raw:
                continue
            h = hashlib.md5(raw).hexdigest()
            path = f"/tmp/{h}.pdf"
            with open(path, "wb") as f:
                f.write(raw)
            tmp_paths.append(path)

            # PDF 로드 (암호화/깨짐 예외 처리)
            try:
                pages = PyPDFLoader(path).load()
            except Exception as e:
                st.warning(f"'{fname}' 로드 실패: {e}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.split_documents(pages)
            for d in docs:
                d.metadata = {**d.metadata, "source_file": fname}
            all_docs.extend(docs)

        if not all_docs:
            raise ValueError("유효한 페이지가 없습니다.")

        vs = FAISS.from_documents(all_docs, OpenAIEmbeddings(model=embed_model))
        return vs, all_docs
    finally:
        # 임시파일 정리
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

# ====== 프롬프트 ======
stuff_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "너는 한국어 금융 전문가야. 아래 '컨텍스트'에서만 근거를 찾아 답해.\n"
        "- 한국어만 사용할 것\n"
        "- 초보자도 이해 가능하게 단계적으로 설명\n"
        "- 수치/전략/위험요소는 구체적으로(기간, 조건 포함)\n"
        "- 컨텍스트에 없으면 '모른다'고 답하고 추측하지 말 것\n"
        "- 과도한 확정 표현 금지\n\n"
        "컨텍스트:\n{context}\n\n"
        "질문: {question}\n"
        "답변:"
    ),
)

# ====== 요약 함수 ======
@st.cache_data(show_spinner=False)
def summarize_once(_docs, _hash):
    _ = str(_hash)
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "다음 텍스트의 핵심을 한국어로 3~5개 불릿으로 요약해.\n"
            "- 수치/위험요소는 구체적으로\n"
            "- 과도한 확정 표현 금지\n\n{text}\n\n요약:"
        ),
    )
    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "아래 요약들을 한국어 단락 3~5문장으로 자연스럽게 통합해. 중복은 제거하고 핵심만 남겨.\n\n{text}\n\n최종 요약:"
        ),
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
    return chain.run(_docs)

# ====== 과거 메시지 렌더 ======
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 토글
do_post = st.toggle("요약/이미지 생성 켜기", key="do_postprocess", value=False)

# ====== 워드클라우드 ======
@st.cache_data(show_spinner=False)
def generate_wordcloud_image_cached(text: str, _hash: str):
    _ = str(_hash)
    stop = set(STOPWORDS) | {"https", "http", "www", "com"}
    wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white", collocations=False, stopwords=stop).generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf

# ====== 메인 입력 ======
if question := st.chat_input("무엇을 도와드릴까요?"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    pdf_mode, docs, retriever, base_retriever = False, None, None, None

    if uploaded_files:
        raws = tuple(f.getvalue() for f in uploaded_files if f is not None)
        names = tuple(f.name for f in uploaded_files if f is not None)
        combined_hash = hashlib.md5(b"".join(raws)).hexdigest() if raws else "nohash"

        try:
            vectorstore, docs = build_vs_multi(raws, names, chunk_size=1000, chunk_overlap=150, embed_model="text-embedding-3-small")
        except Exception as e:
            st.error(f"PDF 처리 오류: {e}")
            pdf_mode = False
        else:
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 12}, search_type="mmr")
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
            pdf_mode = True
            st.session_state["last_file_hash"] = combined_hash
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

    # ====== 답변 생성 ======
    if pdf_mode and retriever:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": stuff_prompt},
            return_source_documents=False,
        )

        response = ""
        try:
            result = qa_chain.invoke({"query": question})
            response = (result.get("result") or "").strip()
        except Exception:
            response = ""

        if not response:
            try:
                r = llm.invoke(f"다음 질문을 문서 검색에 유리하게 한국어로 한 문장으로 재표현해줘: {question}")
                rephrased = getattr(r, "content", str(r)).strip()
            except Exception:
                rephrased = question

            qa_chain_loose = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=base_retriever if base_retriever else retriever,
                chain_type_kwargs={"prompt": stuff_prompt},
                return_source_documents=False,
            )
            try:
                result2 = qa_chain_loose.invoke({"query": rephrased or question})
                response = (result2.get("result") or "").strip()
            except Exception:
                response = ""

        if not response:
            try:
                summary = summarize_once(docs, st.session_state.get("last_file_hash", "nohash"))
                response = "문서에서 직접적인 매칭을 찾기 어려워 요약 기반으로 핵심을 정리했습니다:\n\n" + summary
            except Exception:
                response = "문서에서 관련 내용을 찾기 어렵습니다. 질문을 조금 더 구체화하거나 다른 PDF로 시도해 주세요."

    else:
        prompt = PromptTemplate.from_template(
            """
            너는 금융투자 분야에 특화된 AI야.
            아래 기준으로 한국어로만 간결하게 답해:
            1) 신뢰 가능한 출처 기반  2) 초보자 용어 풀어쓰기  3) 수치/위험요소 포함
            질문: {question}
            """
        )
        try:
            r = llm.invoke(prompt.format(question=question))
            response = getattr(r, "content", str(r))
        except Exception as e:
            response = f"응답 생성 중 오류가 발생했습니다: {e}"

    # ====== 렌더 & 세션 저장 ======
    st.session_state["last_response"] = response
    st.session_state["last_docs"] = docs

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 비동기 로그 저장
    import threading

    def log_async(q, a):
        def _w():
            conn = sqlite3.connect("chat_logs.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT, answer TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute("INSERT INTO chat_logs (question, answer) VALUES (?,?)", (q, a))
            conn.commit()
            conn.close()
        threading.Thread(target=_w, daemon=True).start()

    log_async(question, response)

# ====== 사이드바 ======
with st.sidebar:
    st.markdown("## 📜 투자 FAQ & 가이드")
    with st.expander("📌 자주 묻는 질문"):
        st.markdown(
            """
            **1. ETF란?**  
            상장지수펀드로, 지수를 추종하는 펀드입니다. 주식처럼 거래됩니다.

            **2. ETF와 펀드 차이?**  
            펀드는 하루 1번 거래, ETF는 실시간 거래가 가능합니다.

            **3. 소액으로도 가능한가요?**  
            네. ETF는 1주 단위로 매수할 수 있어, 수천 원으로도 시작할 수 있습니다.

            **4. 수익률은 어떻게 계산하나요?**  
            (현재가 - 매수가) / 매수가 × 100
            """
        )
    with st.expander("💡 초보자를 위한 투자 개념"):
        st.markdown(
            """
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
            """
        )

# ====== 포스트프로세싱 위젯 ======
can_export = bool(st.session_state.get("last_response"))

if st.session_state.get("do_postprocess") and st.session_state.get("last_docs"):
    summary = summarize_once(st.session_state["last_docs"], st.session_state.get("last_file_hash", "nohash"))
    st.success(summary)

    hash_key = st.session_state.get("last_file_hash", "nohash")
    st.image(generate_wordcloud_image_cached(summary, hash_key), use_container_width=True)

# ====== PDF 저장 ======
if st.button("🤖 답변을 PDF로 저장", disabled=not can_export, use_container_width=True):
    try:
        text = st.session_state.get("last_response", "")
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )
        styles = getSampleStyleSheet()
        kstyle = ParagraphStyle(
            "Korean",
            parent=styles["Normal"],
            fontName="NanumGothic",
            fontSize=12,
            leading=16,
            wordWrap="CJK",
        )
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
