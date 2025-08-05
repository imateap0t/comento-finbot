# 베이스 이미지 선택
FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# requirements 먼저 복사해서 설치 (캐시 최적화)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# 포트 설정 (Streamlit 기본 포트)
EXPOSE 8501

# 앱 실행
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]
