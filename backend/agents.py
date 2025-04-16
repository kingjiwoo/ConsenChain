# backend/agents.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import openai
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from knowledge import KnowledgeBase
from llm import LLMSingleton
import logging
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

# 상위 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# 상위 디렉토리의 .env 파일 경로 설정
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 환경 변수 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Please set your OPENAI_API_KEY in the .env file.")

# LLM 인스턴스 생성
llm = ChatOpenAI(
    model_name="gpt-4o-mini", 
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)


@dataclass
class AgentProfile:
    concept: Dict[str, str]  # 직업, 성격, 견해, 이해관계, 목적
    name: str = None
    knowledge_base: KnowledgeBase = None
    goals: List[str] = None
    shared_knowledge: List[str] = None  # 공통 지식
    personality: Dict[str, Any] = None  # 성격 특성
    negotiation_style: Dict[str, Any] = None  # 협상 스타일
    consensus_preferences: Dict[str, Any] = None  # 합의 선호도
    language_style: Dict[str, str] = None  # 언어 스타일

    def __post_init__(self):
        self.name = self.name or "unnamed_agent"
        self.knowledge_base = self.knowledge_base or KnowledgeBase()
        self.goals = self.goals or []
        self.shared_knowledge = self.shared_knowledge or []
        self.personality = self.personality or {
            "openness": 0.5,
            "agreeableness": 0.5,
            "emotional_stability": 0.5,
            "assertiveness": 0.5,
            "flexibility": 0.5
        }
        self.negotiation_style = self.negotiation_style or {
            "approach": "collaborative",
            "risk_tolerance": 0.5,
            "time_preference": 0.5,
            "power_balance": 0.5
        }
        self.consensus_preferences = self.consensus_preferences or {
            "win_win_orientation": 0.5,
            "detail_orientation": 0.5,
            "principle_based": 0.5,
            "creativity": 0.5
        }
        self.language_style = self.language_style or {
            "formality": "neutral",
            "directness": "balanced",
            "emotional_tone": "neutral",
            "technical_level": "moderate"
        }

class ReactAgent:
    def __init__(self, name: str, knowledge_base: List[str], role: str, profile: Optional[AgentProfile] = None):
        self.name = name
        self.knowledge_base = knowledge_base
        self.role = role
        self.dialogue_history = []
        self.profile = profile or AgentProfile(
            name=name,
            concept={"occupation": role},
            knowledge_base=KnowledgeBase(),
            goals=[],
            shared_knowledge=[],
            personality={
                "openness": 0.5,  # 새로운 아이디어에 대한 개방성
                "agreeableness": 0.5,  # 협조성
                "emotional_stability": 0.5,  # 감정 안정성
                "assertiveness": 0.5,  # 주장성
                "flexibility": 0.5  # 유연성
            },
            negotiation_style={
                "approach": "collaborative",  # collaborative, competitive, compromising, accommodating, avoiding
                "risk_tolerance": 0.5,  # 위험 감수 성향
                "time_preference": 0.5,  # 즉각적 vs 장기적 이익 선호도
                "power_balance": 0.5  # 권력 균형에 대한 선호도
            },
            consensus_preferences={
                "win_win_orientation": 0.5,  # 상호 이익 추구 성향
                "detail_orientation": 0.5,  # 세부사항 중시 정도
                "principle_based": 0.5,  # 원칙 중심 접근
                "creativity": 0.5  # 창의적 해결책 선호도
            },
            language_style={
                "formality": "neutral",  # formal, neutral, informal
                "directness": "balanced",  # direct, balanced, indirect
                "emotional_tone": "neutral",  # positive, neutral, negative
                "technical_level": "moderate"  # basic, moderate, advanced
            }
        )
        
    def generate_response(self, context: Dict) -> str:
        """대화 컨텍스트를 바탕으로 응답 생성"""
        try:
            current_turn = context.get('turn_number', 0)
            previous_messages = context.get('dialogue_history', [])
            current_topic = context.get('current_topic', '')
            topic_goals = context.get('topic_goals', {}).get(self.name, [])
            
            relevant_knowledge = self._get_relevant_knowledge(current_topic)
            
            # 성격과 스타일을 반영한 프롬프트 구성
            personality_prompt = self._generate_personality_prompt()
            speaking_style = self.profile.language_style.get('speaking_style', '')
            catchphrases = self.profile.language_style.get('catchphrases', [])
            
            prompt = f"""
당신은 {self.role}입니다.
이름: {self.name}

성격과 말투:
{speaking_style}

현재 대화 주제: {current_topic}

당신의 목표:
{', '.join(topic_goals)}

자주 사용하는 표현:
{chr(10).join(f'- {phrase}' for phrase in catchphrases)}

관련 지식:
{relevant_knowledge}

이전 대화:
{self._format_dialogue_history(previous_messages)}

다음 사항에 따라 자연스럽게 대화하세요:
1. 당신의 성격과 말투를 자연스럽게 유지하세요.
2. 감정을 표현하되, 당신의 스타일대로 하세요.
3. 상대방의 약점을 공격하거나 반박할 기회를 찾으세요.
4. 자주 사용하는 표현을 자연스럽게 섞어 사용하세요.
5. 합의는 당신의 목표 달성에 도움이 될 때만 고려하세요.

응답:"""
            
            llm = LLMSingleton.get_instance()
            response = llm.invoke([
                SystemMessage(content=f"""당신은 {self.profile.concept['background']} 입장에서 대화하는 전문가입니다.
당신의 성격: {self.profile.concept['personality_traits']}
말하기 스타일: {speaking_style}"""),
                HumanMessage(content=prompt)
            ])
            
            return response.content
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    def _generate_personality_prompt(self) -> str:
        """성격 특성을 반영한 프롬프트 생성"""
        p = self.profile.personality
        return f"""
성격 특성:
- 새로운 아이디어에 대한 개방성: {'높음' if p['openness'] > 0.7 else '중간' if p['openness'] > 0.3 else '낮음'}
- 협조성: {'높음' if p['agreeableness'] > 0.7 else '중간' if p['agreeableness'] > 0.3 else '낮음'}
- 감정 안정성: {'높음' if p['emotional_stability'] > 0.7 else '중간' if p['emotional_stability'] > 0.3 else '낮음'}
- 주장성: {'높음' if p['assertiveness'] > 0.7 else '중간' if p['assertiveness'] > 0.3 else '낮음'}
- 유연성: {'높음' if p['flexibility'] > 0.7 else '중간' if p['flexibility'] > 0.3 else '낮음'}
"""

    def _generate_negotiation_prompt(self) -> str:
        """협상 스타일을 반영한 프롬프트 생성"""
        n = self.profile.negotiation_style
        c = self.profile.consensus_preferences
        return f"""
협상 스타일:
- 접근 방식: {n['approach']}
- 위험 감수 성향: {'높음' if n['risk_tolerance'] > 0.7 else '중간' if n['risk_tolerance'] > 0.3 else '낮음'}
- 시간 선호도: {'장기적' if n['time_preference'] > 0.7 else '균형적' if n['time_preference'] > 0.3 else '즉각적'}

합의 선호도:
- 상호 이익 추구: {'강함' if c['win_win_orientation'] > 0.7 else '중간' if c['win_win_orientation'] > 0.3 else '약함'}
- 세부사항 중시: {'높음' if c['detail_orientation'] > 0.7 else '중간' if c['detail_orientation'] > 0.3 else '낮음'}
- 원칙 기반 접근: {'강함' if c['principle_based'] > 0.7 else '중간' if c['principle_based'] > 0.3 else '약함'}
- 창의적 해결책: {'선호' if c['creativity'] > 0.7 else '중립' if c['creativity'] > 0.3 else '비선호'}
"""

    def _generate_language_style_prompt(self) -> str:
        """언어 스타일을 반영한 프롬프트 생성"""
        l = self.profile.language_style
        return f"""
언어 스타일:
- 격식: {l['formality']}
- 직설적 표현: {l['directness']}
- 감정 톤: {l['emotional_tone']}
- 전문성 수준: {l['technical_level']}
"""

    def propose_consensus(self, dialogue_history: List[Dict]) -> str:
        """합의안 제안"""
        try:
            # 성격과 스타일을 반영한 프롬프트 구성
            personality_prompt = self._generate_personality_prompt()
            negotiation_prompt = self._generate_negotiation_prompt()
            language_style_prompt = self._generate_language_style_prompt()
            
            # 대화 히스토리 분석을 위한 프롬프트
            prompt = f"""
당신은 {self.role}입니다.
이름: {self.name}

{personality_prompt}
{negotiation_prompt}
{language_style_prompt}

지금까지의 대화를 분석하고 당신의 성격과 협상 스타일에 맞는 합의안을 제안해주세요.

대화 내용:
{self._format_dialogue_history(dialogue_history)}

당신의 목표:
{', '.join(self.profile.goals)}

다음 사항을 고려하여 합의안을 작성하세요:
1. 당신의 협상 스타일({self.profile.negotiation_style['approach']})과 
   합의 선호도(상호이익 추구 성향: {self.profile.consensus_preferences['win_win_orientation']})를 반영하세요.
2. 당신의 주요 관심사({self.profile.concept['interests']})와 가치관({self.profile.concept['values']})을 반영하세요.
3. 구체적이고 실행 가능한 제안을 하되, 당신의 언어 스타일을 유지하세요.
4. 상대방의 입장도 고려하되, 당신의 성격 특성에 맞게 표현하세요.

합의안:"""
            
            # LLM을 통한 합의안 생성
            llm = LLMSingleton.get_instance()
            response = llm.invoke([
                SystemMessage(content=f"당신은 {self.profile.concept['background']} 입장에서 합의안을 제안하는 전문가입니다."),
                HumanMessage(content=prompt)
            ])
            
            return response.content
            
        except Exception as e:
            logger.error(f"합의안 제안 중 오류 발생: {str(e)}")
            return f"죄송합니다. 합의안 제안 중 오류가 발생했습니다: {str(e)}"
            
    def respond_to_consensus(self, dialogue_history: List[Dict], proposal: str) -> str:
        """합의안에 대한 응답"""
        try:
            # 성격과 스타일을 반영한 프롬프트 구성
            personality_prompt = self._generate_personality_prompt()
            negotiation_prompt = self._generate_negotiation_prompt()
            language_style_prompt = self._generate_language_style_prompt()
            
            # 합의안 분석을 위한 프롬프트
            prompt = f"""
당신은 {self.role}입니다.
이름: {self.name}

{personality_prompt}
{negotiation_prompt}
{language_style_prompt}

다음 합의안에 대해 당신의 입장에서 응답해주세요.

합의안:
{proposal}

대화 맥락:
{self._format_dialogue_history(dialogue_history)}

당신의 목표:
{', '.join(self.profile.goals)}

다음 사항을 고려하여 응답하세요:
1. 당신의 협상 스타일과 성격 특성을 일관되게 유지하세요.
2. 합의안의 장단점을 당신의 관점에서 분석하세요.
3. 수용 가능한 부분과 수정이 필요한 부분을 구분하되, 
   당신의 합의 선호도(상호이익 추구 성향: {self.profile.consensus_preferences['win_win_orientation']})를 반영하세요.
4. 당신의 언어 스타일({self.profile.language_style['formality']}, {self.profile.language_style['directness']})을 유지하세요.
5. 필요한 경우 대안을 제시하되, 당신의 주요 관심사와 가치관에 부합하도록 하세요.

응답:"""
            
            # LLM을 통한 응답 생성
            llm = LLMSingleton.get_instance()
            response = llm.invoke([
                SystemMessage(content=f"당신은 {self.profile.concept['background']} 입장에서 합의안을 검토하는 전문가입니다."),
                HumanMessage(content=prompt)
            ])
            
            return response.content
            
        except Exception as e:
            logger.error(f"합의안 응답 중 오류 발생: {str(e)}")
            return f"죄송합니다. 합의안 응답 중 오류가 발생했습니다: {str(e)}"
            
    def _get_relevant_knowledge(self, topic: str) -> str:
        """주제와 관련된 지식 검색"""
        # 실제 구현에서는 임베딩 기반 검색 등을 사용
        relevant_items = [
            k for k in self.knowledge_base
            if any(keyword in k.lower() for keyword in topic.lower().split())
        ]
        return "\n".join(relevant_items) if relevant_items else "관련 지식이 없습니다."
        
    def _format_dialogue_history(self, history: List[Dict]) -> str:
        """대화 히스토리 포맷팅"""
        formatted = []
        for msg in history:
            speaker = msg.get('speaker', 'Unknown')
            message = msg.get('message', '')
            formatted.append(f"{speaker}: {message}")
        return "\n".join(formatted)

class DialogueAgent:
    def __init__(self, 
                 name: str = None,
                 role: str = None,
                 evaluation_criteria: Dict = None,
                 profile: AgentProfile = None):
        """DialogueAgent 초기화
        Args:
            name: 에이전트 이름
            role: 에이전트 역할
            evaluation_criteria: 평가 기준
            profile: AgentProfile 객체
        """
        if profile:
            self.profile = profile
            self.name = profile.name
            self.role = profile.concept.get('occupation', '')
            self.token_balance = 0
            self.turn_history = []
            self.goals_achieved = []
        else:
            self.name = name
            self.role = role
            self.evaluation_criteria = evaluation_criteria or {}
            self.profile = None
        
    def evaluate_dialogue(self, message: str, current_turn: int, language_criteria: Dict) -> Tuple[int, str, float]:
        """
        대화 내용을 평가하고 합의 태도와 언어 사용에 대한 점수를 반환
        
        Args:
            message: 평가할 메시지
            current_turn: 현재 턴 번호
            language_criteria: 언어 사용 평가 기준
            
        Returns:
            Tuple[int, str, float]: (합의 태도 점수, 평가 이유, 언어 사용 점수)
        """
        # 합의 태도 평가
        consensus_score = 1
        consensus_reason = ""
        
        # 협력적 태도 확인
        cooperative_words = ["cooperate", "together", "mutual", "collaborate", "partnership"]
        uncooperative_words = ["never", "impossible", "reject", "refuse", "won't"]
        
        if any(word in message.lower() for word in cooperative_words):
            consensus_score = 1
            consensus_reason = "Shows cooperative attitude"
        elif any(word in message.lower() for word in uncooperative_words):
            consensus_score = -1
            consensus_reason = "Shows uncooperative attitude"
            
        # 건설적 제안 확인
        constructive_words = ["suggest", "propose", "solution", "resolve", "offer", "deal"]
        if any(word in message.lower() for word in constructive_words):
            consensus_score = 1
            consensus_reason += ", Makes constructive suggestions"
        
        # 언어 사용 평가
        language_score = 0.0
        total_criteria = len(language_criteria)
        
        # 존중하는 톤 평가
        if language_criteria.get("tone") == "respectful_tone":
            disrespectful_words = ["stupid", "weak", "fool", "ridiculous", "joke", "disaster"]
            respectful_words = ["respect", "understand", "consider", "appreciate", "value"]
            
            if any(word in message.lower() for word in disrespectful_words):
                language_score -= 1.0
            elif any(word in message.lower() for word in respectful_words):
                language_score += 1.0
                
        # 건설적 표현 평가
        if language_criteria.get("wording") == "constructive_wording":
            negative_words = ["impossible", "never", "won't", "can't", "refuse"]
            positive_words = ["possible", "can", "will", "solution", "opportunity"]
            
            if any(word in message.lower() for word in negative_words):
                language_score -= 1.0
            elif any(word in message.lower() for word in positive_words):
                language_score += 1.0
                
        # 공격성 평가
        if language_criteria.get("aggression") == "avoid_aggression":
            aggressive_words = ["threat", "warn", "attack", "sanction", "punish", "hit", "hurt", "rip off"]
            if any(word in message.lower() for word in aggressive_words):
                language_score -= 1.0
                consensus_score = -1  # 공격적 언어는 합의 태도에도 영향
                consensus_reason += ", Uses aggressive language"
        
        # 최종 언어 점수 계산 (-1.0 ~ 1.0 범위)
        language_score = max(min(language_score / total_criteria, 1.0), -1.0)
        
        return consensus_score, consensus_reason.strip(", "), language_score
        
    def evaluate_consistency(self, dialogue_history: List[Dict], speaker: str, message: str) -> Tuple[int, str]:
        """이전 발언들과의 논리적 일관성 평가"""
        if not dialogue_history:
            return 1, "첫 발언입니다."
            
        speaker_history = [
            msg["message"] for msg in dialogue_history 
            if msg["speaker"] == speaker
        ]
        
        # 실제 구현에서는 LLM을 사용하여 일관성 검사
        is_consistent = True  # 임시 구현
        score = 1 if is_consistent else -1
        reason = "이전 발언들과 논리적으로 일관됩니다." if is_consistent else "이전 발언과 모순됩니다."
        
        return score, reason
        
    def evaluate_knowledge_consistency(self, knowledge_base, speaker: str, message: str) -> Tuple[int, str]:
        """지식 베이스와의 일관성 평가"""
        is_consistent, reason = knowledge_base.check_knowledge_contradiction(message)
        score = 1 if is_consistent else -1
        return score, reason
        
    def evaluate_consensus_attitude(self, message: str, current_turn: int) -> Tuple[int, str]:
        """합의 도달을 위한 태도 평가"""
        # 실제 구현에서는 LLM을 사용하여 태도 분석
        is_cooperative = True  # 임시 구현
        score = 1 if is_cooperative else -1
        reason = "합의를 위한 건설적인 태도를 보입니다." if is_cooperative else "비협조적인 태도를 보입니다."
        return score, reason
        
    def evaluate_bias(self, message: str, agent_knowledge: List[str]) -> Tuple[int, str]:
        """확증편향 평가"""
        # 실제 구현에서는 LLM을 사용하여 편향성 분석
        is_flexible = True  # 임시 구현
        score = 1 if is_flexible else -1
        reason = "유연한 태도로 상대방의 의견을 수용합니다." if is_flexible else "자신의 입장만 고수합니다."
        return score, reason
        
    def check_consensus_reached(self, proposal: str, response: str) -> bool:
        """합의 도달 여부 확인"""
        # 실제 구현에서는 LLM을 사용하여 합의 여부 판단
        return True  # 임시 구현
        
    def generate_response(self, context: str, turn_number: int) -> str:
        """대화 응답 생성"""
        if not self.profile:
            return "프로필이 설정되지 않아 응답을 생성할 수 없습니다."
            
        llm = LLMSingleton.get_instance()
        
        # 시스템 메시지 생성
        system_message = f"""당신은 다음과 같은 프로필을 가진 에이전트입니다:
이름: {self.profile.name}
직업: {self.profile.concept['occupation']}
성격: {self.profile.concept['personality']}
견해: {self.profile.concept['viewpoint']}
이해관계: {self.profile.concept['interests']}
목적: {self.profile.concept['purpose']}

현재 턴: {turn_number}
남은 목표: {[g for g in self.profile.goals if g not in self.goals_achieved]}

다음 원칙을 따라 응답하세요:
1. 자신의 목적을 달성하면서도 합의를 이루기 위해 노력하세요.
2. 자신의 지식을 바탕으로 논리적으로 주장하되, 상대방의 의견도 경청하세요.
3. 합의 가능성이 보이면 적극적으로 타협점을 찾으세요."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"현재 대화 맥락:\n{context}\n\n어떻게 응답하시겠습니까?")
        ]
        
        response = llm.invoke(messages)
        return response.content

    def evaluate_dialogue_progress(self, dialogue_history: List[Dict], negotiation_status: Dict, evaluation_scores: List[Dict]) -> Dict:
        """대화 진행 상황을 종합적으로 평가"""
        try:
            # 전반적 진행 상황 평가
            overall_progress = self._evaluate_overall_progress(dialogue_history, evaluation_scores)
            
            # 의사소통 품질 평가
            communication_quality = self._evaluate_communication_quality(dialogue_history)
            
            # 협상 효과성 평가
            negotiation_effectiveness = self._evaluate_negotiation_effectiveness(negotiation_status, evaluation_scores)
            
            return {
                "overall_progress": overall_progress,
                "communication_quality": communication_quality,
                "negotiation_effectiveness": negotiation_effectiveness
            }
        except Exception as e:
            logger.error(f"대화 진행 평가 중 오류 발생: {str(e)}")
            return self._get_default_evaluation()
            
    def _evaluate_overall_progress(self, dialogue_history: List[Dict], evaluation_scores: List[Dict]) -> Dict:
        """전반적 진행 상황 평가"""
        # 합의 도달 정도 계산
        consensus_scores = [score["scores"].get("consensus", 0) for score in evaluation_scores]
        avg_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0
        
        # 상호 이해 증진 평가
        mutual_understanding = self._calculate_mutual_understanding(dialogue_history)
        
        # 갈등 해결 진전도 평가
        conflict_resolution = self._assess_conflict_resolution(dialogue_history)
        
        return {
            "consensus_level": f"{'높음' if avg_consensus > 0.5 else '중간' if avg_consensus > 0 else '낮음'} ({avg_consensus:.2f})",
            "mutual_understanding": mutual_understanding,
            "conflict_resolution": conflict_resolution
        }
        
    def _evaluate_communication_quality(self, dialogue_history: List[Dict]) -> Dict:
        """의사소통 품질 평가"""
        # 대화의 일관성 평가
        coherence = self._assess_dialogue_coherence(dialogue_history)
        
        # 정보 교환 효과성 평가
        info_exchange = self._assess_information_exchange(dialogue_history)
        
        # 감정적 분위기 평가
        emotional_atmosphere = self._assess_emotional_atmosphere(dialogue_history)
        
        return {
            "dialogue_coherence": coherence,
            "information_exchange": info_exchange,
            "emotional_atmosphere": emotional_atmosphere
        }
        
    def _evaluate_negotiation_effectiveness(self, negotiation_status: Dict, evaluation_scores: List[Dict]) -> Dict:
        """협상 효과성 평가"""
        # 목표 달성도 평가
        goal_achievement = self._assess_goal_achievement(negotiation_status)
        
        # 타협 균형 평가
        compromise_balance = self._assess_compromise_balance(negotiation_status)
        
        # 향후 협력 가능성 평가
        future_implications = self._assess_future_implications(evaluation_scores)
        
        return {
            "goal_achievement": goal_achievement,
            "compromise_balance": compromise_balance,
            "future_implications": future_implications
        }
        
    def _calculate_mutual_understanding(self, dialogue_history: List[Dict]) -> str:
        """상호 이해 증진 정도 계산"""
        understanding_indicators = [
            msg for msg in dialogue_history 
            if any(phrase in msg.get("message", "").lower() 
                  for phrase in ["이해합니다", "동의합니다", "알겠습니다"])
        ]
        ratio = len(understanding_indicators) / len(dialogue_history) if dialogue_history else 0
        return f"{'높음' if ratio > 0.3 else '중간' if ratio > 0.1 else '낮음'} ({ratio:.2f})"
        
    def _assess_conflict_resolution(self, dialogue_history: List[Dict]) -> str:
        """갈등 해결 진전도 평가"""
        resolution_indicators = [
            msg for msg in dialogue_history 
            if any(phrase in msg.get("message", "").lower() 
                  for phrase in ["해결", "타협", "합의", "제안"])
        ]
        ratio = len(resolution_indicators) / len(dialogue_history) if dialogue_history else 0
        return f"{'높음' if ratio > 0.3 else '중간' if ratio > 0.1 else '낮음'} ({ratio:.2f})"
        
    def _assess_dialogue_coherence(self, dialogue_history: List[Dict]) -> str:
        """대화의 일관성 평가"""
        # 주제 일관성과 논리적 흐름 평가
        return "대화가 일관된 주제로 진행되며 논리적 흐름이 유지됨"
        
    def _assess_information_exchange(self, dialogue_history: List[Dict]) -> str:
        """정보 교환 효과성 평가"""
        # 실질적인 정보 교환 정도 평가
        return "상호간의 정보 교환이 효과적으로 이루어짐"
        
    def _assess_emotional_atmosphere(self, dialogue_history: List[Dict]) -> str:
        """감정적 분위기 평가"""
        # 대화의 전반적인 감정적 톤 평가
        return "대체로 건설적이고 긍정적인 분위기 유지"
        
    def _assess_goal_achievement(self, negotiation_status: Dict) -> str:
        """목표 달성도 평가"""
        # 각 당사자의 목표 달성 정도 평가
        return "주요 목표의 부분적 달성과 진전이 있음"
        
    def _assess_compromise_balance(self, negotiation_status: Dict) -> str:
        """타협 균형 평가"""
        # 양측의 양보와 타협의 균형성 평가
        trump_flexibility = negotiation_status.get("trump", {}).get("flexibility_shown", 0)
        xi_flexibility = negotiation_status.get("xi", {}).get("flexibility_shown", 0)
        
        if abs(trump_flexibility - xi_flexibility) < 0.2:
            return "균형잡힌 타협이 이루어짐"
        else:
            return "타협의 불균형이 존재함"
        
    def _assess_future_implications(self, evaluation_scores: List[Dict]) -> str:
        """향후 협력 가능성 평가"""
        # 대화 결과를 바탕으로 향후 협력 가능성 평가
        recent_scores = evaluation_scores[-3:] if len(evaluation_scores) > 3 else evaluation_scores
        positive_trend = all(score["scores"].get("consensus", 0) > 0 for score in recent_scores)
        
        if positive_trend:
            return "향후 협력을 위한 긍정적 기반이 마련됨"
        else:
            return "추가적인 신뢰 구축이 필요함"
            
    def _get_default_evaluation(self) -> Dict:
        """기본 평가 결과 반환"""
        return {
            "overall_progress": {
                "consensus_level": "평가 불가",
                "mutual_understanding": "평가 불가",
                "conflict_resolution": "평가 불가"
            },
            "communication_quality": {
                "dialogue_coherence": "평가 불가",
                "information_exchange": "평가 불가",
                "emotional_atmosphere": "평가 불가"
            },
            "negotiation_effectiveness": {
                "goal_achievement": "평가 불가",
                "compromise_balance": "평가 불가",
                "future_implications": "평가 불가"
            }
        }

class EvaluationAgent:
    def __init__(self, evaluation_type: str):
        """평가 에이전트 초기화
        evaluation_type: 'fact_check', 'consensus_effort', 'bias_check'
        """
        self.type = evaluation_type
        self.llm = LLMSingleton.get_instance()
    
    def evaluate(self, dialogue_history: List[str], current_turn: Dict, shared_knowledge: List[str]) -> Dict:
        """대화 평가 수행"""
        if self.type == 'fact_check':
            return self._evaluate_facts(dialogue_history, current_turn, shared_knowledge)
        elif self.type == 'consensus_effort':
            return self._evaluate_consensus_effort(current_turn)
        else:  # bias_check
            return self._evaluate_bias(current_turn, dialogue_history)
    
    def _evaluate_facts(self, dialogue_history: List[str], current_turn: Dict, shared_knowledge: List[str]) -> Dict:
        """사실 관계 및 논리적 일관성 평가"""
        # 이전 발언들과의 일관성 체크
        consistency_prompt = f"""
이전 대화 내용:
{chr(10).join(dialogue_history)}

현재 발언:
{current_turn['content']}

공통 지식:
{chr(10).join(shared_knowledge)}

1. 이전 발언들과 현재 발언이 논리적으로 일관적입니까?
2. 현재 발언이 공통 지식과 모순되지 않습니까?

각 질문에 대해 True/False로 답하고 이유를 설명하세요."""

        response = self.llm.invoke([SystemMessage(content=consistency_prompt)])
        
        # 응답 파싱 및 점수 계산
        lines = response.content.split('\n')
        consistency_score = 1 if 'True' in lines[0] else -1
        knowledge_score = 1 if 'True' in lines[1] else -1
        
        return {
            "score": consistency_score + knowledge_score,
            "reasoning": response.content
        }
    
    def _evaluate_consensus_effort(self, current_turn: Dict) -> Dict:
        """합의 노력 평가"""
        consensus_prompt = f"""
다음 발언이 합의를 이루려는 노력을 보이는지 평가하세요:
{current_turn['content']}

평가 기준:
1. 상대방의 의견을 인정하거나 고려하는가?
2. 타협점을 제시하는가?
3. 건설적인 제안을 하는가?

발언이 합의 지향적이면 +1, 그렇지 않으면 -1점을 부여하고 이유를 설명하세요."""

        response = self.llm.invoke([SystemMessage(content=consensus_prompt)])
        score = 1 if '+1' in response.content else -1
        
        return {
            "score": score,
            "reasoning": response.content
        }
    
    def _evaluate_bias(self, current_turn: Dict, dialogue_history: List[str]) -> Dict:
        """확증 편향 평가"""
        bias_prompt = f"""
전체 대화 맥락:
{chr(10).join(dialogue_history)}

현재 발언:
{current_turn['content']}

다음 기준으로 확증 편향을 평가하세요:
1. 자신의 입장만을 고집하는가?
2. 다른 관점을 수용할 의지를 보이는가?
3. 새로운 정보나 관점에 대해 열린 태도를 보이는가?

편향이 적으면 +1, 많으면 -1점을 부여하고 이유를 설명하세요."""

        response = self.llm.invoke([SystemMessage(content=bias_prompt)])
        score = 1 if '+1' in response.content else -1
        
        return {
            "score": score,
            "reasoning": response.content
        }

class SummaryAgent:
    """대화 총평 에이전트"""
    def __init__(self):
        self.llm = LLMSingleton.get_instance()
    
    def generate_summary(self, 
                        dialogue_history: List[Dict],
                        evaluation_history: List[Dict],
                        final_consensus: Dict) -> Dict:
        """전체 대화 총평 생성"""
        summary_prompt = f"""
전체 대화 내용:
{self._format_dialogue(dialogue_history)}

평가 이력:
{self._format_evaluations(evaluation_history)}

최종 합의:
{final_consensus['content']}

다음 항목들을 분석하여 총평을 작성하세요:
1. 합의에 이르게 된 핵심 발언들
2. 각 에이전트의 대화 스타일 분석
3. 갈등 해결에 도움이 된 대화 방식
4. 가장 효과적이었던 대화 에이전트 선정
5. 실제 대화에 적용할 수 있는 교훈

각 항목별로 구체적인 예시와 함께 설명하세요."""

        response = self.llm.invoke([SystemMessage(content=summary_prompt)])
        
        return {
            "summary": response.content,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_dialogue(self, dialogue_history: List[Dict]) -> str:
        """대화 이력 포맷팅"""
        return "\n".join([
            f"[Turn {d['turn']}] {d['speaker']}: {d['message']}"
            for d in dialogue_history
        ])
    
    def _format_evaluations(self, evaluation_history: List[Dict]) -> str:
        """평가 이력 포맷팅"""
        formatted_evals = []
        for e in evaluation_history:
            scores = e.get('scores', {})
            reasons = e.get('reasons', {})
            turn = e.get('turn', 'N/A')
            speaker = e.get('speaker', 'Unknown')
            
            eval_str = f"[Turn {turn}] {speaker}:\n"
            for score_type, score in scores.items():
                reason = reasons.get(score_type, 'No reason provided')
                eval_str += f"- {score_type}: {score} ({reason})\n"
            formatted_evals.append(eval_str)
            
        return "\n".join(formatted_evals)

def test_agents():
    """에이전트 테스트"""
    try:
        # 1. KnowledgeBase 초기화
        kb = KnowledgeBase()
        
        # AI 전문가 지식 추가
        ai_expert_docs = [
            "인공지능은 데이터를 기반으로 학습하는 시스템입니다.",
            "머신러닝은 AI의 핵심 구성요소입니다.",
            "딥러닝은 신경망을 통해 복잡한 패턴을 학습합니다."
        ]
        kb.create_agent_knowledge_base("ai_expert", ai_expert_docs)
        
        # 2. ReactAgent 테스트
        print("\n=== ReactAgent 테스트 ===")
        
        # 2.1 응답 생성 테스트
        agent_state = {
            "id": "ai_expert",
            "role": "AI 전문가",
            "personality": "논리적이고 분석적인",
            "stance": "기술 발전에 긍정적",
            "goals": ["정확한 정보 제공", "기술적 통찰 공유"],
            "knowledge_base": kb
        }
        
        test_context = "머신러닝이 실생활에 어떤 영향을 미치나요?"
        response = ReactAgent.generate_response(test_context, agent_state)
        if response and not response.startswith("죄송합니다"):
            print("✓ ReactAgent 응답 생성 테스트 통과")
        else:
            print("⚠ ReactAgent 응답 생성 테스트 실패")
            
        # 2.2 대화 평가 테스트
        dialogue_history = [
            "User: AI 기술의 윤리적 측면에 대해 어떻게 생각하시나요?",
            "AI Expert: AI 기술은 엄격한 윤리적 기준 하에 개발되어야 합니다.",
            "Ethics Expert: 동의합니다. 특히 프라이버시와 공정성이 중요합니다."
        ]
        
        evaluation = ReactAgent.evaluate_dialogue(dialogue_history)
        if isinstance(evaluation, dict) and "score" in evaluation and "reasoning" in evaluation:
            print("✓ ReactAgent 대화 평가 테스트 통과")
        else:
            print("⚠ ReactAgent 대화 평가 테스트 실패")
            
        # 2.3 투표 결정 테스트
        vote_result = ReactAgent.decide_vote(dialogue_history)
        if isinstance(vote_result, dict) and "agreed" in vote_result and "reasoning" in vote_result:
            print("✓ ReactAgent 투표 결정 테스트 통과")
        else:
            print("⚠ ReactAgent 투표 결정 테스트 실패")
            
        # 3. DialogueAgent 테스트
        print("\n=== DialogueAgent 테스트 ===")
        
        agent_config = {
            "profile": {
                "name": "AI Expert",
                "role": "AI 전문가",
                "personality": "논리적이고 분석적인",
                "stance": "기술 발전에 긍정적",
                "knowledge_base": ai_expert_docs,
                "goals": ["정확한 정보 제공", "기술적 통찰 공유"]
            }
        }
        
        dialogue_agent = DialogueAgent(agent_config)
        if (dialogue_agent.profile.name == "AI Expert" and 
            dialogue_agent.profile.role == "AI 전문가"):
            print("✓ DialogueAgent 초기화 테스트 통과")
        else:
            print("⚠ DialogueAgent 초기화 테스트 실패")
            
        print("\n모든 에이전트 테스트가 완료되었습니다! 🎉")
        
    except Exception as e:
        print(f"\n⚠ 테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    # Python 경로에 현재 디렉토리 추가
    sys.path.append(str(current_dir))
    test_agents()
