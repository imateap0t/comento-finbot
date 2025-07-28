import streamlit as st
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
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# LLM 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 페이지 설정
st.set_page_config(page_title="ETF 챗봇", page_icon="💹")
st.title("💹 ETF 금융 상담 챗봇")
st.markdown("PDF를 업로드하고 질문을 입력하면 답변과 요약을 제공해드립니다.")

# 업로드 및 입력창
uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")
question = st.text_input("궁금한 점을 입력하세요:")

# 로직 실행
if uploaded_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # 프롬프트 템플릿
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

    # 요약
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")
    summary = summarize_chain.run(response)

    # 출력
    st.subheader("🤖 챗봇")
    st.write(response)

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
else:
    st.info("왼쪽에 PDF 파일을 업로드하고 질문을 입력하세요.")
