from typing import Dict, List, Tuple, Any
from typing_extensions import TypedDict
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
import os
from pathlib import Path
from dotenv import load_dotenv

# 상위 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
env_path = parent_dir / '.env'
load_dotenv(dotenv_path=env_path)

class DialogueState(TypedDict, total=False):
    """대화 상태를 저장하는 객체"""
    messages: List[AIMessage]  # 대화 메시지
    agent_states: Dict[str, Dict[str, Any]]  # 에이전트 상태
    evaluations: List[Dict[str, Any]]  # 평가 결과
    votes: List[Dict[str, Any]]  # 투표 결과
    current_speaker: str  # 현재 발화자
    max_turns: int  # 최대 턴 수
    consensus_threshold: float  # 합의 임계값
    next_node: str  # 다음 노드
    current_round: int  # 현재 라운드 (1회 합의 시도 = 5라운드)
    round_speakers: List[str]  # 현재 라운드에서 발언한 에이전트들

def init_dialogue_state(agent_states: Dict) -> Dict:
    """초기 대화 상태 생성"""
    return {
        "messages": [],
        "agent_states": agent_states,
        "evaluations": [],
        "votes": [],
        "current_speaker": list(agent_states.keys())[0],
        "max_turns": 5,  # 1회 합의 시도당 5회의 대화
        "consensus_threshold": 0.75,
        "next_node": "dialogue",
        "current_round": 1,
        "round_speakers": []
    }

def dialogue_node(state: DialogueState, agent_id: str, tools: Dict[str, Tool]) -> DialogueState:
    """대화 노드: 에이전트가 현재 상태를 기반으로 응답을 생성"""
    # 현재 에이전트의 상태 가져오기
    agent_state = state["agent_states"].get(agent_id)
    if not agent_state:
        return state

    # 컨텍스트 생성 (이전 발언들의 맥락 포함)
    context = _create_context(state)
    
    # 에이전트 응답 생성 (이전 발언에 대한 이해와 반응 포함)
    response = tools["generate_response"].run({
        "context": context,
        "agent_state": agent_state,
        "dialogue_round": state["current_round"],
        "previous_messages": state["messages"][-3:] if state["messages"] else []  # 최근 3개 메시지 컨텍스트 제공
    })
    
    # 응답을 상태에 기록
    state["messages"].append(
        AIMessage(content=response, additional_kwargs={
            "agent_id": agent_id,
            "round": state["current_round"]
        })
    )
    state["round_speakers"].append(agent_id)
    
    # 현재 라운드의 모든 에이전트가 발언했는지 확인
    if len(state["round_speakers"]) == len(state["agent_states"]):
        state["current_round"] += 1
        state["round_speakers"] = []
        state["next_node"] = "evaluation" if state["current_round"] > state["max_turns"] else "dialogue"
    else:
        state["next_node"] = "dialogue"
    
    state["current_speaker"] = _select_next_speaker(state)
    return state

def evaluation_node(state: DialogueState, evaluator_id: str, tools: Dict[str, Tool]) -> DialogueState:
    """평가 노드: 현재까지의 대화를 평가"""
    # 평가 수행
    evaluation = tools["evaluate_dialogue"].run({
        "dialogue_history": [msg.content for msg in state["messages"]],
        "evaluator_id": evaluator_id
    })
    
    # 평가 결과를 상태에 기록
    state["evaluations"].append({
        "evaluator": evaluator_id,
        "result": evaluation
    })
    
    # 다음 노드 설정
    state["next_node"] = "consensus_vote"
    
    return state

def consensus_vote_node(state: DialogueState, agent_id: str, tools: Dict[str, Tool]) -> DialogueState:
    """합의 투표 노드: 현재 상태에 대한 에이전트의 동의 여부 결정"""
    # 투표 결정
    vote_result = tools["decide_vote"].run({
        "dialogue_history": [msg.content for msg in state["messages"]],
        "agent_id": agent_id,
        "agent_state": state["agent_states"][agent_id]
    })
    
    # 투표 결과를 상태에 기록
    state["votes"].append({
        "agent_id": agent_id,
        "agreed": vote_result["agreed"],
        "reasoning": vote_result["reasoning"]
    })
    
    # 다음 노드 설정
    state["next_node"] = "should_continue"
    
    return state

def should_continue_node(state: DialogueState) -> Tuple[str, bool]:
    """다음 단계 결정 노드: 대화 계속 또는 종료 결정"""
    # 합의 도달 여부 확인
    if len(state["votes"]) >= len(state["agent_states"]):
        agreed_count = sum(1 for vote in state["votes"] if vote["agreed"])
        consensus_reached = (agreed_count / len(state["votes"])) >= state["consensus_threshold"]
        
        if consensus_reached:
            state["next_node"] = "end"
            return "end", True
    
    # 최대 턴 수 확인
    if len(state["messages"]) >= state["max_turns"]:
        state["next_node"] = "end"
        return "end", True
        
    # 다음 발화자 선택
    next_speaker = _select_next_speaker(state)
    state["next_node"] = "dialogue"
    return next_speaker, False

def _create_context(state: DialogueState) -> str:
    """현재 대화 컨텍스트 생성 - 이전 발언들의 맥락 포함"""
    context_parts = []
    
    # 라운드 정보 추가
    context_parts.append(f"=== 현재 라운드: {state['current_round']}/{state['max_turns']} ===\n")
    
    # 이전 발언들의 맥락 추가
    recent_messages = state["messages"][-3:] if state["messages"] else []
    for msg in recent_messages:
        speaker = msg.additional_kwargs['agent_id']
        round_num = msg.additional_kwargs['round']
        content = msg.content
        context_parts.append(f"[라운드 {round_num}] {speaker}: {content}")
    
    return "\n".join(context_parts)

def _select_next_speaker(state: DialogueState) -> str:
    """다음 발화자 선택 - 현재 라운드에서 아직 발언하지 않은 에이전트 중에서 선택"""
    available_agents = [
        agent_id for agent_id in state["agent_states"].keys()
        if agent_id not in state["round_speakers"]
    ]
    return available_agents[0] if available_agents else list(state["agent_states"].keys())[0]

def test_dialogue_state():
    """DialogueState 테스트"""
    print("\n=== DialogueState 테스트 시작 ===")
    
    # 초기 상태 생성
    initial_agent_states = {
        "ai_expert": {
            "name": "AI 전문가",
            "role": "기술 전문가",
            "personality": "논리적이고 분석적인",
            "stance": "기술 발전에 긍정적",
            "goals": ["정확한 정보 제공", "기술적 통찰 공유"]
        },
        "ethics_expert": {
            "name": "윤리 전문가",
            "role": "윤리 연구원",
            "personality": "신중하고 균형잡힌",
            "stance": "윤리적 고려 중시",
            "goals": ["윤리적 관점 제시", "사회적 영향 평가"]
        }
    }
    
    state = init_dialogue_state(initial_agent_states)
    
    # 테스트용 도구
    test_tools = {
        "generate_response": Tool(
            name="generate_response",
            func=lambda x: f"{x['agent_state']['role']}: AI 기술의 발전은 신중하게 접근해야 합니다.",
            description="Generate a response based on context and agent state"
        ),
        "evaluate_dialogue": Tool(
            name="evaluate_dialogue",
            func=lambda x: {"score": 0.8, "reasoning": "논리적 일관성이 높습니다."},
            description="Evaluate dialogue based on specific criteria"
        ),
        "decide_vote": Tool(
            name="decide_vote",
            func=lambda x: {"agreed": True, "reasoning": "윤리적 고려가 충분합니다."},
            description="Decide vote based on dialogue history and agent state"
        )
    }
    
    try:
        # 1. 초기 상태 테스트
        print("\n1. 초기 상태 테스트")
        if (
            isinstance(state, dict) and
            len(state["messages"]) == 0 and
            state["next_node"] == "dialogue"
        ):
            print("✓ 초기 상태 테스트 통과")
            print(f"  현재 상태: {state['next_node']}")
        else:
            print("⚠ 초기 상태 테스트 실패")
        
        # 2. 대화 노드 테스트
        print("\n2. 대화 노드 테스트")
        state = dialogue_node(state, "ai_expert", test_tools)
        if (
            len(state["messages"]) == 1 and
            state["next_node"] == "evaluation"
        ):
            print("✓ 대화 노드 테스트 통과")
            print(f"  메시지: {state['messages'][-1].content}")
            print(f"  다음 노드: {state['next_node']}")
        else:
            print("⚠ 대화 노드 테스트 실패")
        
        # 3. 평가 노드 테스트
        print("\n3. 평가 노드 테스트")
        state = evaluation_node(state, "logic", test_tools)
        if (
            len(state["evaluations"]) == 1 and
            state["next_node"] == "consensus_vote"
        ):
            print("✓ 평가 노드 테스트 통과")
            print(f"  평가 결과: {state['evaluations'][-1]}")
            print(f"  다음 노드: {state['next_node']}")
        else:
            print("⚠ 평가 노드 테스트 실패")
        
        # 4. 합의 투표 노드 테스트
        print("\n4. 합의 투표 노드 테스트")
        state = consensus_vote_node(state, "ai_expert", test_tools)
        if (
            len(state["votes"]) == 1 and
            state["next_node"] == "should_continue"
        ):
            print("✓ 합의 투표 노드 테스트 통과")
            print(f"  투표 결과: {state['votes'][-1]}")
            print(f"  다음 노드: {state['next_node']}")
        else:
            print("⚠ 합의 투표 노드 테스트 실패")
        
        # 5. 흐름 제어 노드 테스트
        print("\n5. 흐름 제어 노드 테스트")
        next_speaker, should_end = should_continue_node(state)
        if (
            isinstance(next_speaker, str) and
            isinstance(should_end, bool) and
            state["next_node"] in ["dialogue", "end"]
        ):
            print("✓ 흐름 제어 노드 테스트 통과")
            print(f"  다음 발화자: {next_speaker}")
            print(f"  종료 여부: {should_end}")
            print(f"  다음 노드: {state['next_node']}")
        else:
            print("⚠ 흐름 제어 노드 테스트 실패")
        
        print("\n모든 DialogueState 테스트가 완료되었습니다! 🎉")
        
    except Exception as e:
        print(f"\n⚠ 테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_dialogue_state()
