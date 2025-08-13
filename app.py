import hashlib
import os
import sqlite3
import time
import threading
from io import BytesIO

import streamlit as st
from datetime import datetime
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
from langchain_core.callbacks import BaseCallbackHandler
from langchain.retrievers import EnsembleRetriever

try:
    from langchain_community.retrievers import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# ====== PDF/시각화 ======
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from wordcloud import WordCloud, STOPWORDS

# ====== 기본 설정 ======
load_dotenv()

st.set_page_config(page_title="금융 챗봇", page_icon="💹")

def measure_response_time(func):
    """응답 시간 측정 함수"""
    start_time = time.time()
    
    try:
        result = func()
        success = True
    except Exception as e:
        success = False
        st.session_state.performance_metrics["errors"] += 1
        raise e
    finally:
        end_time = time.time()
        response_time = end_time - start_time
        
        # 성능 메트릭 업데이트
        st.session_state.performance_metrics["query_count"] += 1
        st.session_state.performance_metrics["total_time"] += response_time
        st.session_state.performance_metrics["response_times"].append({
            "time": response_time,
            "timestamp": datetime.now(),
            "success": success
        })
        
        # 실시간 성능 표시
        avg_time = st.session_state.performance_metrics["total_time"] / st.session_state.performance_metrics["query_count"]
        
        if response_time > 10:
            st.warning(f"⚠️ 응답시간: {response_time:.2f}초 (평균: {avg_time:.2f}초)")
        else:
            st.info(f"⚡ 응답시간: {response_time:.2f}초 (평균: {avg_time:.2f}초)")
    
    return result

if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = {
        "query_count": 0,
        "total_time": 0,
        "response_times": [],
        "errors": 0
    }

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("""
    🚨 **OpenAI API 키가 설정되지 않았습니다**
    
    `.env` 파일에 다음과 같이 추가해주세요:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```
    
    또는 Streamlit Cloud에서 Secrets를 설정해주세요.
    """)
    st.stop()

# LLM 설정
try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)
    # API 키 유효성
    test_response = llm.invoke("test")
except Exception as e:
    st.error(f"🚨 OpenAI API 연결 실패: {str(e)}")
    if "api_key" in str(e).lower():
        st.error("API 키가 유효하지 않습니다. 키를 확인해주세요.")
    st.stop()

# UI 헤더
st.title("💹 금융 상담 챗봇")
st.markdown("금융에 대해 무엇이든 물어보세요!")

# 한글 폰트 등록
font_path = os.path.join(os.path.dirname(__file__), "NanumGothic-Regular.ttf")
pdfmetrics.registerFont(TTFont("NanumGothic", font_path))

# CSS
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
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    embed_model: str = "text-embedding-3-small",
):
    # 캐시 키 안정화: 파일 해시 + 파라미터
    _ = (chunk_size, chunk_overlap, embed_model, tuple(hashlib.md5(b).hexdigest() for b in files))

    all_docs = []
    tmp_paths = []
    processed_files = []  # 성공적으로 처리된 파일 목록
    
    try:
        for idx, (raw, fname) in enumerate(zip(files, file_names)):
            if not raw:
                st.warning(f"⚠️ '{fname}': 파일이 비어있습니다.")
                continue
                
            h = hashlib.md5(raw).hexdigest()
            path = f"/tmp/{h}.pdf"
            
            try:
                with open(path, "wb") as f:
                    f.write(raw)
                tmp_paths.append(path)

                # PDF 로드 (암호화/깨짐 예외 처리)
                pages = PyPDFLoader(path).load()
                if not pages:
                    st.warning(f"⚠️ '{fname}': 문서가 비어있습니다.")
                    continue
                    
                total_text = "".join(page.page_content for page in pages)
                if len(total_text.strip()) < 30:
                    st.warning(f"⚠️ '{fname}': 텍스트 추출이 어렵습니다.")
                    continue

                # 청크 분할
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                docs = splitter.split_documents(pages)
                
                # 각 청크에 파일 정보 추가
                for d in docs:
                    d.metadata = {
                        **d.metadata, 
                        "source_file": fname,
                        "file_index": idx,  # 파일 순서 추가
                        "total_files": len(files)  # 전체 파일 수 추가
                    }
                
                all_docs.extend(docs)
                processed_files.append(fname)
                st.success(f"✅ '{fname}' 처리 완료 ({len(docs)}개 청크)")

            except Exception as e:
                error_msg = str(e).lower()
                if "encrypted" in error_msg or "password" in error_msg:
                    st.warning(f"🔒 '{fname}': 암호화된 PDF입니다. 암호를 해제하고 다시 업로드해주세요.")
                elif "damaged" in error_msg or "corrupted" in error_msg:
                    st.warning(f"💥 '{fname}': 손상된 파일입니다.")
                elif "permission" in error_msg:
                    st.warning(f"🚫 '{fname}': 파일 접근 권한이 없습니다.")
                else:
                    st.warning(f"❌ '{fname}' 로드 실패: {e}")
                continue

        if not all_docs:
            raise ValueError("유효한 페이지가 없습니다.")

        # 처리 결과 요약 표시
        st.info(f"📄 총 {len(processed_files)}개 파일에서 {len(all_docs)}개 청크를 생성했습니다.")
        
        # 파일별 청크 수 표시
        file_chunk_counts = {}
        for doc in all_docs:
            fname = doc.metadata.get("source_file", "Unknown")
            file_chunk_counts[fname] = file_chunk_counts.get(fname, 0) + 1
        
        for fname, count in file_chunk_counts.items():
            st.write(f"  • {fname}: {count}개 청크")

        try:
            vs = FAISS.from_documents(all_docs, OpenAIEmbeddings(model=embed_model))
            st.success(f"✅ {len(all_docs)}개의 문서 청크로 벡터 데이터베이스를 구성했습니다.")
            
        except Exception as e:
            st.error(f"🚨 벡터 데이터베이스 생성 실패: {e}")
            if "api" in str(e).lower():
                st.error("OpenAI API 호출 중 오류가 발생했습니다. API 키와 네트워크를 확인해주세요.")
            raise e
        
        return vs, all_docs

    finally:
        # 임시파일 정리
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    #show_file_processing_status(processed_files, len(files))

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
        "**멀티 문서 처리 중요 규칙**:\n"
        "1. 컨텍스트에 여러 문서의 정보가 포함되어 있으면 반드시 모든 문서를 종합하여 답변\n"
        "2. 각 문서의 고유한 특징이나 차이점이 있으면 명확히 구분하여 설명\n"
        "3. 문서간 공통점과 차이점을 비교 분석하여 제시\n"
        "4. 답변 시 '첫 번째 문서에서는...', '두 번째 문서에서는...' 식으로 출처를 명시\n"
        "5. 질문이 전체적인 비교나 종합을 요구하면 모든 문서를 아우르는 답변 제공\n\n"
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

# ====== 스트리밍 콜백 핸들러 =======
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_start(self, *args, **kwargs):
        self.text = ""
        if self.container:
            self.container.markdown("")
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        if self.container:
            self.container.markdown(self.text)
    def on_llm_end(self, *args, **kwargs):
        pass

def log_async(q, a):
        """비동기로 대화 로그를 데이터베이스에 저장"""
        def _w():
            try:
                conn = sqlite3.connect("chat_logs.db", check_same_thread=False)
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT, 
                        answer TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute("INSERT INTO chat_logs (question, answer) VALUES (?,?)", (q, a))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Log save error: {e}")
        
        threading.Thread(target=_w, daemon=True).start()


# ====== 과거 메시지 렌더 ======
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 토글
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ 추가 기능")
    
    # 요약/이미지 생성 토글
    do_post = st.toggle(
        "📊 요약/워드클라우드 자동 생성", 
        key="do_postprocess", 
        value=False,
        help="PDF 문서가 업로드된 상태에서 답변 시 자동으로 문서 요약과 워드클라우드를 생성합니다."
    )
    
    # 스트리밍 출력 토글
    stream_on = st.toggle(
        "⚡ 실시간 스트리밍", 
        key="do_stream", 
        value=True,
        help="답변을 실시간으로 표시합니다. 끄면 완성된 답변을 한 번에 보여줍니다."
    )
    
    # 토글 상태 표시
    if do_post:
        if st.session_state.get("last_docs"):
            st.success("✅ 다음 답변부터 요약/워드클라우드가 생성됩니다")
        else:
            st.info("ℹ️ PDF를 업로드하고 질문하면 요약/워드클라우드가 생성됩니다")

# ====== 워드클라우드 ======
def debug_search_results(sources, question):
    """검색 결과를 분석하여 멀티 파일 처리 상태를 확인"""
    if not sources:
        st.warning("⚠️ 검색 결과가 없습니다.")
        return
    
    file_counts = {}
    total_chars = 0
    
    for doc in sources:
        source_file = doc.metadata.get("source_file", "Unknown")
        file_counts[source_file] = file_counts.get(source_file, 0) + 1
        total_chars += len(doc.page_content)
    
    with st.expander("🔍 검색 디버그 정보", expanded=False):
        st.write(f"• 총 검색된 청크 수: {len(sources)}")
        st.write(f"• 총 텍스트 길이: {total_chars:,} 문자")
        
        for file_name, count in file_counts.items():
            percentage = (count / len(sources)) * 100
            st.write(f"• {file_name}: {count}개 청크 ({percentage:.1f}%)")
        
        # 파일별 균형 검사
        if len(file_counts) > 1:
            counts = list(file_counts.values())
            if max(counts) / min(counts) > 3:
                st.warning("⚠️ 파일간 검색 결과 불균형 감지!")

@st.cache_data(show_spinner=False)
def generate_wordcloud_image_cached(text: str, _hash: str):
    """워드클라우드 이미지 생성 (캐시됨)"""
    _ = str(_hash)
    
    # 한글 불용어 확장
    korean_stopwords = {
        "그리고", "하지만", "그러나", "또한", "따라서", "그래서", "즉", "예를 들어",
        "있습니다", "입니다", "합니다", "됩니다", "것입니다", "수", "때", "등", "통해",
        "대한", "위한", "관련", "경우", "방법", "이런", "그런", "이와", "같은"
    }
    
    stop = set(STOPWORDS) | {"https", "http", "www", "com"} | korean_stopwords

    try:
        wc = WordCloud(
            font_path=font_path, 
            width=800, 
            height=400, 
            background_color="white", 
            collocations=False, 
            stopwords=stop,
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        return buf
        
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류: {e}")
        return None

# ====== 메인 입력 ======
if question := st.chat_input("무엇을 도와드릴까요?"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 화면 자리 + 콜백
    live_area = st.empty()
    handler = StreamHandler(live_area) if st.session_state.get("do_stream", False) else None
    cfg = {"callbacks": [handler]} if handler else {}

    st.session_state["last_sources"] = []
    pdf_mode, docs, retriever, base_retriever = False, None, None, None

    if uploaded_files:
        raws = tuple(f.getvalue() for f in uploaded_files if f is not None)
        names = tuple(f.name for f in uploaded_files if f is not None)
        combined_hash = hashlib.md5(b"".join(raws)).hexdigest() if raws else "nohash"

        pdf_mode, docs, retriever, base_retriever = False, None, None, None

        try:
            with st.spinner("📄 PDF 문서를 처리하고 있습니다..."):
                vectorstore, docs = build_vs_multi(
                    raws, names, 
                    chunk_size=1000, 
                    chunk_overlap=150, 
                    embed_model="text-embedding-3-small"
                )
        except ValueError as e:
            st.error(f"📄 문서 처리 오류: {e}")
            st.info("💡 다른 PDF 파일을 업로드하거나, 텍스트가 포함된 PDF인지 확인해주세요.")
            pdf_mode = False
        except Exception as e:
            st.error(f"🚨 예상치 못한 오류가 발생했습니다: {e}")
            st.info("💡 파일 크기가 너무 크거나 네트워크 문제일 수 있습니다. 잠시 후 다시 시도해주세요.")
            pdf_mode = False
        else:
            # Dense(임베딩) 검색 설정
            dense = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.3})
            
            # BM25 사용 가능 여부 확인 및 조건부 사용
            try:
                from langchain_community.retrievers import BM25Retriever
                bm25 = BM25Retriever.from_documents(docs)
                ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
                st.success("🔍 하이브리드 검색 활성화")
            except ImportError:
                # BM25 라이브러리가 없는 경우
                ensemble = dense
                st.info("🔍 Dense 검색 활성화 (BM25 라이브러리 미설치)")
            except Exception as e:
                # BM25 초기화 실패한 경우
                ensemble = dense
                st.info("🔍 Dense 검색 활성화 (BM25 초기화 실패)")

            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

            base_retriever = dense
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
            return_source_documents=True,
        )

        response = ""
        sources = []
        try:
            with st.spinner("🔍 문서에서 답변을 찾고 있습니다..."):
                res = measure_response_time(
                    lambda: qa_chain.invoke({"query": question}, config=cfg)
                )
                response = (res.get("result") or "").strip()
                sources = res.get("source_documents", []) or []
                
                # 검색 결과 디버깅 (개발용)
                if st.session_state.get("debug_mode", False):
                    debug_search_results(sources, question)
                
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                st.error("⏳ API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
                response = "잠시 후 다시 질문해주세요."
            elif "timeout" in error_msg:
                st.error("⏱️ 응답 시간이 초과되었습니다. 질문을 더 간단하게 해보세요.")
                response = "질문을 더 간단하게 해주시겠어요?"
            elif "token" in error_msg:
                st.error("📝 문서가 너무 길어 처리할 수 없습니다. 더 작은 PDF로 시도해보세요.")
                response = "문서가 너무 큽니다. 더 작은 문서로 시도해주세요."
            else:
                st.error(f"🚨 답변 생성 중 오류: {e}")
                response = "죄송합니다. 일시적인 오류가 발생했습니다."
            sources = []

        # ====== 렌더 & 세션 저장 ======
        st.session_state["last_response"] = response
        if pdf_mode and docs:
            st.session_state["last_docs"] = docs
            st.session_state["last_file_hash"] = combined_hash
        else:
            st.session_state.pop("last_docs", None)

        try:
            if handler:
                live_area.empty()
        except Exception:
            pass


        # 소스 전처리 후 세션 저장
        processed_sources = []
        _seen = set()
        file_sources = {}

        for d in sources:
            src_name = d.metadata.get("source_file") or os.path.basename(d.metadata.get("source", "?"))
            pg = d.metadata.get("page")
            pg_num = (pg + 1) if isinstance(pg, int) else "?"

            # 파일별로 그룹화
            if src_name not in file_sources:
                file_sources[src_name] = []

            key = (src_name, pg_num)
            if key in _seen:
                continue
            _seen.add(key)

            snippet = (d.page_content or "").strip()
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            
            file_sources[src_name].append({"page": pg_num, "snippet": snippet})

        # 파일별로 정리된 소스 정보 생성
        for fname, pages in file_sources.items():
            for page_info in pages:
                processed_sources.append({
                    "file": fname, 
                    "page": page_info["page"], 
                    "snippet": page_info["snippet"]
                })

        st.session_state["last_sources"] = processed_sources

        # 답변 표시 부분에서 파일별로 그룹화해서 보여주기:
        if st.session_state.get("last_sources"):
            with st.expander("🔎 참고 문서"):
                # 파일별로 그룹화해서 표시
                current_sources = st.session_state["last_sources"]
                file_groups = {}
                
                for source in current_sources:
                    fname = source["file"]
                    if fname not in file_groups:
                        file_groups[fname] = []
                    file_groups[fname].append(source)
                
                for fname, sources_in_file in file_groups.items():
                    st.markdown(f"**📁 {fname}**")
                    for i, s in enumerate(sources_in_file, 1):
                        st.markdown(f"  {i}. p.{s['page']}")
                        st.code(s["snippet"], language="text")
                    st.markdown("---")

        # 첫 번째 검색에서 답변이 없는 경우 백업 검색
        if not response or len(response.strip()) < 50 or "모른다" in response:
            st.info("🔍 추가 검색을 시도합니다...")
            
            # 백업 검색 체인 생성
            qa_chain_loose = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff", 
                retriever=base_retriever,
                chain_type_kwargs={"prompt": stuff_prompt},
                return_source_documents=True,
            )
            
            # 단순화된 백업 검색
            backup_queries = [
                f"핵심 키워드: {question}",
                f"간단한 질문: {question}",
                question  # 원본 질문
            ]

            for backup_query in backup_queries:
                try:
                    backup_result = qa_chain_loose.invoke({"query": backup_query})
                    backup_response = (backup_result.get("result") or "").strip()
                    
                    if backup_response and "모른다" not in backup_response and len(backup_response) > 50:
                        response = backup_response
                        if backup_query != question:
                            st.info(f"💡 '{backup_query}'로 재검색하여 답변을 찾았습니다.")
                        break
                except:
                    continue

            # 여전히 답변이 없으면 키워드 기반 마지막 시도
            if not response or len(response.strip()) < 20:
                try:
                    keyword_prompt = f"다음 질문에서 가장 중요한 키워드 3개만 추출해줘 (쉼표로 구분): {question}"
                    keyword_result = llm.invoke(keyword_prompt)
                    keywords = getattr(keyword_result, "content", "").strip()
                    
                    if keywords:
                        result3 = qa_chain_loose.invoke({"query": keywords}, config=cfg)
                        test_response = (result3.get("result") or "").strip()
                        
                        if test_response and "모른다" not in test_response:
                            response = f"키워드 '{keywords}' 기반 검색 결과:\n\n{test_response}"
                            st.info(f"🔍 키워드 검색: '{keywords}'")
                            
                except Exception:
                    pass

        qa_chain_loose = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=base_retriever if base_retriever else retriever,
            chain_type_kwargs={"prompt": stuff_prompt},
            return_source_documents=True,
        )
        try:
            result2 = qa_chain_loose.invoke({"query": rephrased or question}, config=cfg)
            response = (result2.get("result") or "").strip()
        except Exception:
            response = ""

        if not response:
            try:
                # 질문에서 키워드 추출
                keyword_prompt = f"다음 질문에서 가장 중요한 키워드 3개만 추출해줘 (쉼표로 구분): {question}"
                keyword_result = llm.invoke(keyword_prompt)
                keywords = getattr(keyword_result, "content", "").strip()
                
                # 키워드로 다시 검색
                if keywords:
                    result3 = qa_chain_loose.invoke({"query": keywords}, config=cfg)
                    test_response = (result3.get("result") or "").strip()
                    
                    if test_response and "모른다" not in test_response:
                        response = f"키워드 '{keywords}' 기반 검색 결과:\n\n{test_response}"
                        
            except Exception:
                pass
    else:
        prompt = PromptTemplate.from_template(
            template=(
                "너는 한국어 금융 전문가야. 일반적인 금융 지식을 바탕으로 답변해줘.\n"
                "- 한국어만 사용할 것\n"
                "- 초보자도 이해 가능하게 단계적으로 설명\n"
                "- 수치/전략/위험요소는 구체적으로(기간, 조건 포함)\n"
                "- 유용한 일반적 정보를 포함해도 됨\n\n"
                "질문: {question}\n"
                "답변:"
            )
        )

        try:
            with st.spinner("💭 답변을 생성하고 있습니다..."):
                r = llm.invoke(prompt.format(context="", question=question), config=cfg)
                response = getattr(r, "content", str(r))
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                response = "⏳ 요청이 많아 잠시 대기 중입니다. 1분 후 다시 시도해주세요."
                st.warning("API 요청 한도 초과. 잠시 후 다시 시도해주세요.")
            elif "api_key" in error_msg:
                response = "🔒 API 키 문제가 발생했습니다. 관리자에게 문의해주세요."
                st.error("API 키 오류 발생")
            else:
                response = f"죄송합니다. 일시적인 오류로 답변을 생성할 수 없습니다. 다시 시도해주세요."
                st.error(f"응답 생성 오류: {e}")

    
    # ====== 답변 표시 ======
    with st.chat_message("assistant"):
        st.markdown(response)

        # 참고 문서 표시 (파일별로 그룹화)
        if st.session_state.get("last_sources"):
            with st.expander("🔎 참고 문서"):
                current_sources = st.session_state["last_sources"]
                file_groups = {}
                
                # 파일별로 그룹화
                for source in current_sources:
                    fname = source["file"]
                    if fname not in file_groups:
                        file_groups[fname] = []
                    file_groups[fname].append(source)
                
                # 파일별로 표시
                for idx, (fname, sources_in_file) in enumerate(file_groups.items()):
                    st.markdown(f"**📁 {fname}**")
                    for i, s in enumerate(sources_in_file, 1):
                        st.markdown(f"  {i}. p.{s['page']}")
                        st.code(s["snippet"], language="text")

                    if idx < len(file_groups) - 1:
                        st.markdown("---")

        # 요약/워드클라우드 기능
        if st.session_state.get("do_postprocess"):
            if st.session_state.get("last_docs"):
                st.markdown("---")
                st.subheader("📊 문서 분석")
                
                try:
                    with st.spinner("📝 문서를 분석하고 있습니다..."):
                        summary = summarize_once(
                            st.session_state["last_docs"], 
                            st.session_state.get("last_file_hash", "nohash")
                        )
                    
                    with st.expander("📋 문서 요약", expanded=True):
                        st.info(summary)
                    
                    try:
                        with st.spinner("🎨 키워드를 시각화하고 있습니다..."):
                            hash_key = st.session_state.get("last_file_hash", "nohash")
                            wordcloud_img = generate_wordcloud_image_cached(summary, hash_key)
                        
                        if wordcloud_img:
                            with st.expander("☁️ 키워드 클라우드", expanded=True):
                                st.image(wordcloud_img, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"워드클라우드 생성 실패: {e}")
                        
                except Exception as e:
                    st.error(f"문서 분석 실패: {e}")
            
            elif len(response) > 100:
                st.markdown("---")
                st.subheader("💡 답변 키워드")
                
                try:
                    simple_wordcloud = generate_wordcloud_image_cached(response, "response_based")
                    if simple_wordcloud:
                        with st.expander("☁️ 답변 키워드", expanded=False):
                            st.image(simple_wordcloud, use_container_width=True)
                            st.caption("현재 답변에서 추출한 주요 키워드들입니다.")
                
                except Exception as e:
                    st.warning(f"키워드 분석 실패: {e}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # 비동기 로그 저장
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

# can_export 변수 정의
can_export = bool(st.session_state.get("last_response"))

with st.sidebar:
    st.markdown("---")
    st.subheader("🔄 대화 관리")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 삭제", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("last_response", None)
            st.session_state.pop("last_sources", None)
            st.rerun()
    
    with col2:
        if st.button("💾 대화 저장", use_container_width=True):
            if st.session_state.messages:
                # 대화를 텍스트로 변환
                conversation = ""
                for msg in st.session_state.messages:
                    role = "사용자" if msg["role"] == "user" else "챗봇"
                    conversation += f"{role}: {msg['content']}\n\n"
                
                st.download_button(
                    "📄 대화 내역 다운로드",
                    data=conversation.encode('utf-8'),
                    file_name=f"대화기록_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_chat"
                )

# 성능 위젯
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 개발자 도구")
    
    debug_mode = st.toggle(
        "🐛 검색 디버그", 
        key="debug_mode", 
        value=False,
        help="검색 결과의 파일별 분포를 확인합니다."
    )

with st.sidebar:
    st.markdown("---")
    st.subheader("📊 실시간 성능")
    
    metrics = st.session_state.performance_metrics
    
    if metrics["query_count"] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "총 쿼리", 
                metrics["query_count"],
                help="이번 세션에서 처리한 총 질문 수"
            )
            
        with col2:
            avg_time = metrics["total_time"] / metrics["query_count"]
            st.metric(
                "평균 응답시간", 
                f"{avg_time:.1f}초",
                help="평균적인 응답 생성 시간"
            )
        
        # 성공률 표시
        success_rate = ((metrics["query_count"] - metrics["errors"]) / metrics["query_count"]) * 100
        st.metric("성공률", f"{success_rate:.1f}%")
        
        # 최근 응답시간 차트 (간단한 라인)
        if len(metrics["response_times"]) > 1:
            recent_times = [r["time"] for r in metrics["response_times"][-10:]]
            st.line_chart(recent_times)
            st.caption("최근 10개 응답시간 추이")
    else:
        st.info("아직 성능 데이터가 없습니다.")


# ====== 포스트프로세싱 위젯 ======

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
