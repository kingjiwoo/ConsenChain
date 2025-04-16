# backend/graph.py

import sys
import json
from typing import List, Dict, Any, Tuple, TypedDict, Annotated
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from nodes import (
    dialogue_node,
    evaluation_node,
    consensus_vote_node,
    should_continue_node,
    DialogueState
)
from uuid import uuid4
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 노드 함수들을 백엔드/nodes.py에서 불러온다고 가정합니다.

class DialogueState(TypedDict, total=False):
    """대화 상태를 정의하는 타입"""
    messages: List[Dict[str, str]]
    current_turn: int
    consensus_reached: bool
    evaluation_scores: List[Dict[str, Any]]
    knowledge_base: Any
    agents: List[Dict[str, str]]
    config: Dict[str, Any]
    thread_id: str
    votes: List[Dict[str, Any]]

class DialogueGraph:
    def __init__(self, tools: Dict[str, Tool]):
        """대화 그래프 초기화"""
        self.tools = tools
        self.graph = StateGraph(state_schema=DialogueState)
        
        # 노드 설정
        self.graph.add_node("generate_response", self._generate_response)
        self.graph.add_node("evaluate_dialogue", self._evaluate_dialogue)
        self.graph.add_node("decide_vote", self._decide_vote)
        
        # 엣지 설정
        self.graph.add_edge("generate_response", "evaluate_dialogue")
        self.graph.add_edge("evaluate_dialogue", "decide_vote")
        self.graph.add_conditional_edges(
            "decide_vote",
            self._should_continue,
            {
                True: "generate_response",
                False: END
            }
        )
        
        # 시작 노드 설정
        self.graph.set_entry_point("generate_response")
        
    def invoke(self, initial_state: Dict, config: Dict = None) -> Dict:
        """대화 그래프 실행"""
        if config is None:
            config = {}
        
        # 그래프 컴파일
        app = self.graph.compile()
        
        # 실행 설정
        run_config = {
            "max_turns": config.get('max_turns', 5),
            "consensus_threshold": config.get('consensus_threshold', 0.75),
            "evaluation_interval": config.get('evaluation_interval', 2)
        }
        
        # 상태 초기화
        state = {
            "messages": initial_state.get("messages", []),
            "current_turn": initial_state.get("current_turn", 0),
            "consensus_reached": initial_state.get("consensus_reached", False),
            "evaluation_scores": initial_state.get("evaluation_scores", []),
            "knowledge_base": initial_state.get("knowledge_base"),
            "agents": initial_state.get("agents", []),
            "config": run_config,
            "thread_id": "1"  # 고정된 thread_id 사용
        }
        
        # 그래프 실행
        final_state = app.invoke(state)
        return final_state
        
    def _generate_response(self, state: Dict) -> Dict:
        """응답 생성 노드"""
        try:
            current_turn = state.get("current_turn", 0)
            messages = state.get("messages", [])
            
            # 현재 에이전트 선택
            agent_idx = current_turn % len(state["agents"])
            current_agent = state["agents"][agent_idx]
            
            # 컨텍스트 생성
            context = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in messages[-3:]])
            logger.info(f"\n=== 현재 대화 컨텍스트 ===\n{context}")
            
            # 응답 생성을 위한 에이전트 상태 구성
            agent_state = {
                "id": current_agent["id"],
                "role": current_agent["role"],
                "personality": current_agent["personality"],
                "stance": current_agent["stance"],
                "knowledge_base": state["knowledge_base"],
                "goals": ["정확한 정보 제공", "합의 도출"] if current_agent["id"] == "ai_expert" else ["윤리적 고려사항 제시", "균형잡힌 접근"]
            }
            
            # 응답 생성
            response = self.tools["generate_response"].func(
                context=context,
                agent_state=agent_state
            )
            logger.info(f"\n=== {current_agent['role']}의 응답 ===\n{response}")
            
            # 메시지 추가
            messages.append({
                "speaker": current_agent["role"],
                "content": response
            })
            
            return {
                **state,
                "messages": messages,
                "current_turn": current_turn + 1
            }
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            raise Exception(f"응답 생성 중 오류 발생: {str(e)}")
            
    def _evaluate_dialogue(self, state: Dict) -> Dict:
        """대화 평가 노드"""
        try:
            current_turn = state.get("current_turn", 0)
            evaluation_interval = state["config"]["evaluation_interval"]
            
            # 평가 주기 확인
            if current_turn % evaluation_interval != 0:
                return state
                
            messages = state.get("messages", [])
            dialogue_history = [f"{msg['speaker']}: {msg['content']}" for msg in messages]
            
            # 평가 수행
            evaluation = self.tools["evaluate_dialogue"].func(
                dialogue_history=dialogue_history,
                evaluator_id="logic"
            )
            
            # 평가 결과 저장
            evaluation_scores = state.get("evaluation_scores", [])
            evaluation_scores.append(evaluation)
            
            return {
                **state,
                "evaluation_scores": evaluation_scores
            }
            
        except Exception as e:
            raise Exception(f"대화 평가 중 오류 발생: {str(e)}")
            
    def _decide_vote(self, state: Dict) -> Dict:
        """투표 결정 노드"""
        try:
            messages = state.get("messages", [])
            dialogue_history = [f"{msg['speaker']}: {msg['content']}" for msg in messages]
            
            # 각 에이전트의 투표 수집
            votes = []
            for agent in state["agents"]:
                vote = self.tools["decide_vote"].func(
                    dialogue_history=dialogue_history,
                    agent_id=agent["id"],
                    agent_state={
                        "role": agent["role"],
                        "knowledge_base": state["knowledge_base"]
                    }
                )
                votes.append(vote)
            
            # 합의 여부 확인
            agree_count = sum(1 for v in votes if v["agreed"])
            consensus_reached = (agree_count / len(votes)) >= state["config"]["consensus_threshold"]
            
            return {
                **state,
                "consensus_reached": consensus_reached,
                "votes": votes
            }
            
        except Exception as e:
            raise Exception(f"투표 결정 중 오류 발생: {str(e)}")
            
    def _should_continue(self, state: Dict) -> bool:
        """계속 진행 여부 결정"""
        max_turns = state["config"]["max_turns"]
        current_turn = state.get("current_turn", 0)
        consensus_reached = state.get("consensus_reached", False)
        
        return current_turn < max_turns and not consensus_reached

def test_dialogue_graph():
    """DialogueGraph 테스트"""
    try:
        print("\n=== DialogueGraph 테스트 ===")
        
        # 1. 그래프 초기화 테스트
        tools = {
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
        
        graph = DialogueGraph(tools)
        if graph and graph.graph:
            print("✓ 그래프 초기화 테스트 통과")
        else:
            print("⚠ 그래프 초기화 테스트 실패")
            
        # 2. 노드 구성 테스트
        expected_nodes = {"generate_response", "evaluate_dialogue", "decide_vote"}
        actual_nodes = set(graph.graph.nodes.keys())
        if expected_nodes.issubset(actual_nodes):
            print("✓ 노드 구성 테스트 통과")
            print(f"  구성된 노드: {actual_nodes}")
        else:
            print("⚠ 노드 구성 테스트 실패")
            print(f"  기대한 노드: {expected_nodes}")
            print(f"  실제 노드: {actual_nodes}")
            
        # 3. 상태 관리 테스트
        config = {
            "agents": [
                {
                    "id": "ai_expert",
                    "name": "AI Expert",
                    "role": "AI 전문가",
                    "personality": "논리적",
                    "stance": "기술 긍정",
                    "goals": ["정확한 정보 제공"]
                },
                {
                    "id": "ethics_expert",
                    "name": "Ethics Expert",
                    "role": "윤리 전문가",
                    "personality": "신중함",
                    "stance": "윤리 중시",
                    "goals": ["윤리적 고려사항 제시"]
                }
            ],
            "max_turns": 10,
            "consensus_threshold": 0.75
        }
        
        try:
            result = graph.invoke(config)
            print("✓ 상태 관리 테스트 통과")
            print(f"  최종 상태: {result['next_node']}")
        except Exception as e:
            print("⚠ 상태 관리 테스트 실패")
            print(f"  오류: {str(e)}")
        
        print("\n모든 DialogueGraph 테스트가 완료되었습니다! 🎉")
        
    except Exception as e:
        print(f"\n⚠ 테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_dialogue_graph()
