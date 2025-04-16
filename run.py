import uvicorn
from dotenv import load_dotenv
import os

def main():
    # 환경 변수 로드
    load_dotenv()
    
    # 서버 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # 서버 실행
    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main() 