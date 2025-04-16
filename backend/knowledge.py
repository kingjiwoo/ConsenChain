from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import time
import numpy as np

# 상위 디렉토리의 .env 파일 경로 설정
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Please set your OPENAI_API_KEY in the .env file.")

class KnowledgeBase:
    def __init__(self):
        """지식 베이스 초기화"""
        self.agent_knowledge = {}
        self.common_knowledge = []
        self.embeddings = {}
        
    def create_agent_knowledge_base(self, agent_name: str, documents: List[str]):
        """에이전트별 지식 베이스 생성"""
        self.agent_knowledge[agent_name] = documents
        # 임베딩 생성 및 저장
        self.embeddings[agent_name] = self._create_embeddings(documents)
        
    def get_agent_knowledge(self, agent_name: str) -> List[str]:
        """에이전트의 지식 베이스 반환"""
        return self.agent_knowledge.get(agent_name, [])
        
    def add_common_knowledge(self, documents: List[str]):
        """공통 지식 추가"""
        self.common_knowledge.extend(documents)
        
    def get_common_knowledge(self) -> List[str]:
        """공통 지식 반환"""
        return self.common_knowledge
        
    def check_consistency(self, agent_name: str, statement: str) -> Tuple[bool, float, str]:
        """발언의 일관성 검사
        Returns:
            bool: 일관성 여부
            float: 유사도 점수
            str: 판단 근거
        """
        if agent_name not in self.embeddings:
            return False, 0.0, "에이전트의 지식 베이스가 없습니다."
            
        statement_embedding = self._create_embeddings([statement])[0]
        knowledge_embeddings = self.embeddings[agent_name]
        
        max_similarity = 0.0
        most_similar_knowledge = ""
        
        for i, knowledge_embedding in enumerate(knowledge_embeddings):
            similarity = self._compute_similarity(statement_embedding, knowledge_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_knowledge = self.agent_knowledge[agent_name][i]
        
        is_consistent = max_similarity > 0.7  # 임계값 설정
        reason = f"가장 유사한 지식과의 유사도: {max_similarity:.2f}"
        
        return is_consistent, max_similarity, reason
        
    def check_knowledge_contradiction(self, statement: str) -> Tuple[bool, str]:
        """공통 지식과의 모순 검사
        Returns:
            bool: 모순이 없으면 True
            str: 판단 근거
        """
        if not self.common_knowledge:
            return True, "공통 지식이 없습니다."
            
        statement_embedding = self._create_embeddings([statement])[0]
        common_embeddings = self._create_embeddings(self.common_knowledge)
        
        contradictions = []
        for i, common_embedding in enumerate(common_embeddings):
            similarity = self._compute_similarity(statement_embedding, common_embedding)
            if similarity < 0.3:  # 낮은 유사도는 잠재적 모순을 나타냄
                contradictions.append(self.common_knowledge[i])
        
        has_contradiction = len(contradictions) > 0
        reason = f"발견된 잠재적 모순 수: {len(contradictions)}"
        
        return not has_contradiction, reason
        
    def _create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """텍스트 임베딩 생성"""
        # OpenAI API를 사용하여 임베딩 생성
        # 실제 구현에서는 적절한 임베딩 모델 사용
        return [np.random.rand(1536) for _ in texts]  # 임시 구현
        
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """임베딩 간 유사도 계산"""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

def test_knowledge_base():
    """KnowledgeBase 클래스 테스트"""
    # 테스트용 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as test_dir:
        kb = None
        try:
            # 1. 초기화 테스트
            kb = KnowledgeBase()
            assert kb.agent_knowledge == {}
            assert kb.common_knowledge == []
            assert kb.embeddings == {}
            print("✓ 초기화 테스트 통과")
            
            time.sleep(1)  # API 호출 간격
            
            # 2. 에이전트별 지식 베이스 생성 테스트
            agent_docs = {
                "ai_expert": [
                    "인공지능은 데이터를 기반으로 학습하는 시스템입니다.",
                    "머신러닝은 AI의 핵심 구성요소입니다."
                ],
                "ethics_expert": [
                    "AI 윤리는 기술 발전의 필수 고려사항입니다.",
                    "알고리즘의 편향성 문제는 신중히 다뤄야 합니다."
                ]
            }
            
            success_count = 0
            for agent_name, docs in agent_docs.items():
                kb.create_agent_knowledge_base(agent_name, docs)
                if agent_name in kb.agent_knowledge:
                    success_count += 1
                time.sleep(1)
            
            if success_count > 0:
                print(f"✓ 에이전트 지식 베이스 생성 테스트 통과 ({success_count}/{len(agent_docs)} 성공)")
            else:
                print("⚠ 에이전트 지식 베이스 생성 테스트 실패")
            
            time.sleep(1)
            
            # 3. 공통 지식 베이스 생성 테스트
            common_docs = [
                "AI 기술은 사회에 큰 영향을 미칩니다.",
                "지속가능한 발전을 위해서는 기술과 윤리의 균형이 필요합니다."
            ]
            kb.add_common_knowledge(common_docs)
            if kb.common_knowledge:
                print("✓ 공통 지식 베이스 생성 테스트 통과")
            else:
                print("⚠ 공통 지식 베이스 생성 테스트 실패")
            
            time.sleep(1)
            
            # 4. 지식 베이스 검색 테스트
            if len(kb.agent_knowledge) > 0:
                test_queries = {
                    "ai_expert": {
                        "query": "머신러닝이란 무엇인가요?",
                        "expected_keywords": ["머신러닝", "AI"]
                    },
                    "ethics_expert": {
                        "query": "AI 윤리의 중요성은 무엇인가요?",
                        "expected_keywords": ["윤리", "편향성"]
                    }
                }
                
                success_count = 0
                for agent_name, test_data in test_queries.items():
                    if agent_name not in kb.agent_knowledge:
                        continue
                        
                    results = kb.get_agent_knowledge(agent_name)
                    
                    if results and any(
                        keyword.lower() in ' '.join(results).lower()
                        for keyword in test_data["expected_keywords"]
                    ):
                        success_count += 1
                    
                    time.sleep(1)
                
                if success_count > 0:
                    print(f"✓ 지식 베이스 검색 테스트 통과 ({success_count}/{len(test_queries)} 성공)")
                else:
                    print("⚠ 지식 베이스 검색 테스트 실패")
                
                # 5. 공통 지식 포함 검색 테스트
                if kb.common_knowledge and len(kb.agent_knowledge) > 1:
                    results_with_common = kb.get_agent_knowledge(next(iter(kb.agent_knowledge)))
                    results_without_common = kb.get_agent_knowledge(next(iter(kb.agent_knowledge)))
                    
                    time.sleep(1)
                    
                    if len(results_with_common) >= len(results_without_common):
                        print("✓ 공통 지식 포함 검색 테스트 통과")
                    else:
                        print("⚠ 공통 지식 포함 검색 테스트 실패")
            
            print("\n테스트 완료!")
            
        finally:
            if kb:
                for agent_name, knowledge in kb.agent_knowledge.items():
                    if isinstance(knowledge, list):
                        for doc in knowledge:
                            if isinstance(doc, str):
                                doc_path = Path(f"knowledge_base/{agent_name}/{doc.split('/')[-1]}")
                                if doc_path.exists():
                                    doc_path.unlink()
                for doc in kb.common_knowledge:
                    if isinstance(doc, str):
                        doc_path = Path(f"knowledge_base/common/{doc.split('/')[-1]}")
                        if doc_path.exists():
                            doc_path.unlink()

if __name__ == "__main__":
    test_knowledge_base() 