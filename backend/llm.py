from langchain_openai import ChatOpenAI
import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# 상위 디렉토리의 .env 파일 경로 설정
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class LLMSingleton:
    _instance: Optional[ChatOpenAI] = None
    _evaluation_instance: Optional[ChatOpenAI] = None
    
    @classmethod
    def get_instance(cls, for_evaluation: bool = False) -> ChatOpenAI:
        """LLM 인스턴스 반환"""
        if for_evaluation:
            if cls._evaluation_instance is None:
                cls._evaluation_instance = ChatOpenAI(
                    temperature=0.3,
                    model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
                    request_timeout=float(os.getenv("REQUEST_TIMEOUT", "30")),
                    max_retries=int(os.getenv("MAX_RETRIES", "3"))
                )
            return cls._evaluation_instance
            
        if cls._instance is None:
            cls._instance = ChatOpenAI(
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                model_name=os.getenv("MODEL_NAME", "gpt-4"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
                request_timeout=float(os.getenv("REQUEST_TIMEOUT", "30")),
                max_retries=int(os.getenv("MAX_RETRIES", "3"))
            )
        return cls._instance 

if __name__ == "__main__":
    from langchain.schema import HumanMessage, SystemMessage
    
    def test_llm_singleton():
        """LLM 싱글톤 패턴 테스트"""
        # 일반 대화용 LLM 테스트
        llm1 = LLMSingleton.get_instance()
        llm2 = LLMSingleton.get_instance()
        assert llm1 is llm2, "일반 대화용 LLM 인스턴스가 동일하지 않습니다."
        
        # 평가용 LLM 테스트
        eval_llm1 = LLMSingleton.get_instance(for_evaluation=True)
        eval_llm2 = LLMSingleton.get_instance(for_evaluation=True)
        assert eval_llm1 is eval_llm2, "평가용 LLM 인스턴스가 동일하지 않습니다."
        
        # 일반 대화용과 평가용이 다른 인스턴스인지 확인
        assert llm1 is not eval_llm1, "일반 대화용과 평가용 LLM이 같은 인스턴스입니다."
        print("싱글톤 패턴 테스트 통과!")

    def test_llm_response():
        """LLM 응답 생성 테스트"""
        llm = LLMSingleton.get_instance()
        
        # 시스템 메시지와 사용자 메시지로 구성된 간단한 대화 테스트
        messages = [
            SystemMessage(content="당신은 AI 윤리 전문가입니다."),
            HumanMessage(content="AI 윤리에서 가장 중요한 것은 무엇인가요?")
        ]
        
        response = llm.invoke(messages)
        assert response is not None, "LLM 응답이 None입니다."
        assert isinstance(response.content, str), "응답 내용이 문자열이 아닙니다."
        assert len(response.content) > 0, "응답 내용이 비어있습니다."
        print("응답 생성 테스트 통과!")

    def test_evaluation_llm():
        """평가용 LLM 테스트"""
        eval_llm = LLMSingleton.get_instance(for_evaluation=True)
        
        # 평가 시나리오 테스트
        messages = [
            SystemMessage(content="당신은 대화의 논리성을 평가하는 평가자입니다."),
            HumanMessage(content="""
            다음 대화를 평가해주세요:
            A: AI는 투명성이 중요합니다.
            B: 네, 하지만 성능도 중요하죠.
            """)
        ]
        
        response = eval_llm.invoke(messages)
        assert response is not None, "평가용 LLM 응답이 None입니다."
        assert isinstance(response.content, str), "평가용 응답 내용이 문자열이 아닙니다."
        assert len(response.content) > 0, "평가용 응답 내용이 비어있습니다."
        print("평가용 LLM 테스트 통과!")

    # 테스트 실행
    print("\nLLM 모듈 테스트 시작...")
    test_llm_singleton()
    test_llm_response()
    test_evaluation_llm()
    print("모든 LLM 테스트가 성공적으로 완료되었습니다!") 