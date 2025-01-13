# Get Started

#### 요구 사항 (Requirements)

- **Python**: 3.9.0
- **pip**: 20.2.3

#### 1. 가상환경 생성

python -m venv .venv  
.venv/Scripts/Activate 로 가상환경으로 전환

#### 2. 모듈 설치

pip install -r requirements.txt

#### 3. 환경변수 .env 파일 생성

LANGCHAIN_TRACING_V2=true  
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"  
LANGCHAIN_API_KEY="랭체인키"  
LANGCHAIN_PROJECT="pr-prickly-evidence-9"  
SERPER_API_KEY="서퍼 api키"

#### 4. ollama Custom Model 생성

Modelfile에 gguf파일 경로 설정  
PowerShell에 ollama create 모델 이름 -f ./Modelfile

#### 5. Streamlit 실행

git bash로 sh run.sh 실행
