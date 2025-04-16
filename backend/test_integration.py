import os
from dotenv import load_dotenv
import time
from typing import Dict, List
import logging
from datetime import datetime

from knowledge import KnowledgeBase
from blockchain import BlockchainManager
from agents import ReactAgent, DialogueAgent, AgentProfile, SummaryAgent, EvaluationAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_test_environment():
    """테스트 환경 설정"""
    # .env 파일 로드
    load_dotenv()
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # 테스트용 설정
    config = {
        'rpc_url': 'http://localhost:8545',  # 테스트용 로컬 블록체인
        'contract_address': '0x0000000000000000000000000000000000000000',
        'use_web3': False  # 로컬 테스트를 위해 Web3 비활성화
    }
    
    return config

def create_test_knowledge_base():
    """테스트용 지식 베이스 생성"""
    knowledge_base = KnowledgeBase()
    
    # 트럼프 스타일 전문가 지식 베이스
    trump_expert_docs = [
        "미국 우선주의의는 글로벌 무역에서 가장 중요한 원칙입니다.",
        "중국에 대한 강력한 관세 정책은 미국 경제를 보호하는 유일한 방법입니다.",
        "무역 적자는 미국이 다른 나라들에게 이용당하고 있다는 명백한 증거입니다.",
        "관세는 미국 기업과 근로자를 보호하는 강력한 무기입니다.",
        "중국의 불공정 무역 관행은 세계 경제 질서를 파괴하고 있습니다.",
        "강력한 무역 제재만이 중국의 행동을 변화시킬 수 있습니다."
    ]
    knowledge_base.create_agent_knowledge_base("trump_expert", trump_expert_docs)
    
    # 시진핑 스타일 전문가 지식 베이스
    xi_expert_docs = [
        "상호 이익과 협력은 글로벌 경제 발전의 기초입니다.",
        "미국의 일방적인 관세 정책은 세계 경제 질서를 훼손하고 있습니다.",
        "중국의 경제 성장은 전 세계의 발전에 기여하고 있습니다.",
        "보호무역주의는 글로벌 공급망을 파괴하고 모든 국가에 피해를 줍니다.",
        "일대일로 정책은 세계 경제의 평화로운 발전을 촉진합니다.",
        "무역 전쟁은 양국 모두에게 해로우며, 대화와 협상이 필요합니다."
    ]
    knowledge_base.create_agent_knowledge_base("xi_expert", xi_expert_docs)
    
    return knowledge_base

def print_detailed_results(blockchain_manager, context: Dict):
    """대화 결과 상세 분석 출력
    
    Args:
        blockchain_manager: BlockchainManager 인스턴스
        context: 테스트 컨텍스트 정보를 담은 딕셔너리
    """
    logger.info("\n=== 상세 테스트 결과 ===")
    
    # 대화 통계
    logger.info("\n1. 대화 통계")
    for agent in ["trump_expert", "xi_expert"]:
        dialogue_count = blockchain_manager.speaker_counts.get(agent, 0)
        logger.info(f"{agent} 발언 횟수: {dialogue_count}")
    
    # 토큰 통계
    logger.info("\n2. 토큰 통계")
    for agent in ["trump_expert", "xi_expert"]:
        token_status = blockchain_manager.get_agent_status(agent)
        logger.info(f"\n{agent} 토큰 현황:")
        logger.info(f"- 현재 잔액: {token_status['token_balance']} CST")
        logger.info("- 최근 거래:")
        for tx in token_status.get('token_history', [])[-3:]:  # 최근 3개 거래만 표시
            logger.info(f"  * {tx['amount']} 토큰: {tx['reason']}")
    
    # 블록체인 상태
    logger.info("\n3. 블록체인 상태")
    chain_length = len(blockchain_manager.blockchain.chain)
    total_transactions = sum(len(block.transactions) 
                           for block in blockchain_manager.blockchain.chain)
    logger.info(f"- 총 블록 수: {chain_length}")
    logger.info(f"- 총 트랜잭션 수: {total_transactions}")
    logger.info(f"- 마지막 블록 해시: {blockchain_manager.blockchain.get_latest_block().hash}")
    
    # 합의 통계
    logger.info("\n4. 합의 통계")
    for agent in ["trump_expert", "xi_expert"]:
        consensus_stats = blockchain_manager.blockchain.get_consensus_statistics(agent)
        logger.info(f"\n{agent} 합의 참여:")
        logger.info(f"- 총 참여 횟수: {consensus_stats['total_participations']}")
        logger.info(f"- 성공한 합의: {consensus_stats['successful_consensus']}")
        logger.info(f"- 실패한 합의: {consensus_stats['failed_consensus']}")

def verify_dialogue_balance(blockchain_manager):
    """대화 참여 균형 검증"""
    dialogue_history = blockchain_manager.get_dialogue_history()
    speaker_counts = {}
    
    # 발언 횟수 집계
    for dialogue in dialogue_history:
        speaker = dialogue["speaker"]
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
    # 균형 상태 평가
    if speaker_counts:
        max_count = max(speaker_counts.values())
        min_count = min(speaker_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 1
        
        print("\n=== 대화 참여 균형 검증 ===")
        for speaker, count in speaker_counts.items():
            print(f"{speaker}: {count}회 발언")
        print(f"균형 비율: {balance_ratio:.2f}")
        
        return balance_ratio >= 0.5  # 최소 50% 이상의 균형 필요
    return True

def verify_token_distribution(blockchain_manager: BlockchainManager) -> bool:
    """토큰 분배의 적절성을 검증
    
    Args:
        blockchain_manager: 블록체인 매니저 인스턴스
        
    Returns:
        bool: 토큰 분배가 적절한지 여부
    """
    # 토큰 분배 이력 조회
    distribution_history = blockchain_manager.get_token_distribution_history()
    
    if not distribution_history:
        logger.warning("토큰 분배 이력이 없습니다.")
        return True
        
    # 마지막 분배 상태 확인
    latest_distribution = distribution_history[-1]["balances"]
    
    # 토큰 보유량 분석
    balances = list(latest_distribution.values())
    if not balances:
        logger.warning("토큰 보유 정보가 없습니다.")
        return True
        
    max_balance = max(balances)
    min_balance = min(balances)
    avg_balance = sum(balances) / len(balances)
    
    # 결과 로깅
    logger.info("\n=== 토큰 분배 현황 ===")
    for agent, balance in latest_distribution.items():
        logger.info(f"{agent}: {balance} 토큰")
    logger.info(f"\n최대 보유량: {max_balance}")
    logger.info(f"최소 보유량: {min_balance}")
    logger.info(f"평균 보유량: {avg_balance:.2f}")
    
    # 토큰 분배의 균형성 검증
    # 최대 보유량과 최소 보유량의 차이가 평균의 50%를 넘지 않아야 함
    balance_threshold = avg_balance * 0.5
    is_balanced = (max_balance - min_balance) <= balance_threshold
    
    if not is_balanced:
        logger.warning("토큰 분배가 불균형합니다.")
        logger.warning(f"최대-최소 차이: {max_balance - min_balance}")
        logger.warning(f"허용 임계값: {balance_threshold}")
    
    return is_balanced

def verify_evaluation_impact(blockchain_manager):
    """평가가 토큰 분배에 미치는 영향 검증"""
    evaluation_history = blockchain_manager.get_evaluation_history()
    
    print("\n=== 평가 영향 검증 ===")
    for eval in evaluation_history:
        print(f"\n평가자: {eval['evaluator']}")
        print(f"대상: {eval['subject']}")
        print(f"점수: {eval['score']}")
        print(f"근거: {eval['reasoning']}")
        print(f"턴 번호: {eval.get('turn_number', 'N/A')}")
        
    # 평가 기록이 있는지 확인
    if not evaluation_history:
        logger.warning("평가 기록이 없습니다.")
        return False
        
    # 각 평가가 토큰 분배에 영향을 미쳤는지 확인
    for eval in evaluation_history:
        token_history = blockchain_manager.get_transaction_history(eval['subject'])
        evaluation_related_transactions = [
            tx for tx in token_history 
            if tx['reason'] and ('evaluation' in tx['reason'].lower() or 'quality' in tx['reason'].lower())
        ]
        
        if not evaluation_related_transactions:
            logger.warning(f"{eval['subject']}에 대한 평가 보상 기록이 없습니다.")
            return False
    
    return True

def verify_consensus_process(blockchain_manager) -> bool:
    """합의 프로세스 검증"""
    # 대화 참여자들의 합의 참여 기록 확인
    for agent in ["trump_expert", "xi_expert"]:
        consensus_stats = blockchain_manager.blockchain.get_consensus_statistics(agent)
        logger.info(f"{agent}의 합의 통계:")
        logger.info(f"- 총 참여: {consensus_stats['total_participations']}")
        logger.info(f"- 성공한 합의: {consensus_stats['successful_consensus']}")
        logger.info(f"- 실패한 합의: {consensus_stats['failed_consensus']}")
        
        # 최소한의 합의 참여가 있었는지 확인
        if consensus_stats["total_participations"] == 0:
            logger.warning(f"{agent}의 합의 참여 기록이 없습니다.")
            return False
            
    # 마지막 턴에서의 합의 보상 확인
    for agent in ["trump_expert", "xi_expert"]:
        token_history = blockchain_manager.get_transaction_history(agent)
        consensus_rewards = [
            tx for tx in token_history 
            if tx['reason'] and 'consensus' in tx['reason'].lower()
        ]
        
        if not consensus_rewards:
            logger.warning(f"{agent}에 대한 합의 보상 기록이 없습니다.")
            return False
    
    return True

def verify_mining_rewards(blockchain_manager) -> bool:
    """채굴 보상 검증"""
    total_blocks = len(blockchain_manager.blockchain.chain)
    if total_blocks <= 1:  # 제네시스 블록만 있는 경우
        logger.warning("생성된 블록이 없습니다.")
        return False
        
    # 채굴된 블록 수 확인
    mined_blocks = total_blocks - 1  # 제네시스 블록 제외
    logger.info(f"총 채굴된 블록 수: {mined_blocks}")
    
    # 각 블록의 트랜잭션 확인
    for block in blockchain_manager.blockchain.chain[1:]:  # 제네시스 블록 제외
        if not block.transactions:
            logger.warning(f"빈 블록이 발견되었습니다: {block.hash}")
            return False
    return True

def test_dialogue_flow() -> bool:
    try:
        # 환경 설정
        config = setup_test_environment()
        
        # BlockchainManager 초기화
        blockchain_manager = BlockchainManager(
            difficulty=4,
            token_rewards={
                "participation": 5,
                "evaluation": 3,
                "quality": 10,
                "consensus": 5
            }
        )
        
        # 지식 베이스 생성
        knowledge_base = create_test_knowledge_base()
        
        # 평가 에이전트 초기화
        fact_checker = EvaluationAgent('fact_check')
        consensus_evaluator = EvaluationAgent('consensus_effort')
        bias_evaluator = EvaluationAgent('bias_check')
        
        # 테스트 컨텍스트 설정
        context = {
            "turn_number": 1,
            "total_turns": 3,
            "dialogue_agents": ["trump_expert", "xi_expert"],
            "evaluation_agents": {
                "fact_checker": fact_checker,
                "consensus_evaluator": consensus_evaluator,
                "bias_evaluator": bias_evaluator
            },
            "knowledge_base": knowledge_base
        }
        
        # 에이전트 초기화 및 초기 토큰 할당
        for agent_id in context["dialogue_agents"]:
            blockchain_manager.initialize_agent(agent_id, initial_tokens=100)
        
        # 공통 지식 설정
        shared_knowledge = [
            "미중 무역 통계는 공식적으로 기록되고 있습니다.",
            "양국은 WTO 회원국으로서 국제 무역 규범을 준수해야 합니다.",
            "지적재산권 보호는 국제 무역의 중요한 원칙입니다.",
            "무역 협상은 상호 이익을 추구하는 방향으로 진행되어야 합니다."
        ]
        
        # 대화 히스토리 저장
        dialogue_history = []
        
        # 미리 정의된 대화 내용
        trump_dialogues = [
            "미국은 중국과의 무역에서 수십 년간 손해를 보고 있습니다. 통계를 보면 매년 수천억 달러의 무역적자가 발생하고 있어요. 강력한 관세 정책으로 이를 바로잡아야 합니다!",
            "중국의 불공정 무역으로 인해 미국 기업들이 피해를 보고 있습니다. 지적재산권 침해와 기술 탈취가 계속되고 있어요. 우리는 더 이상 이를 용납할 수 없습니다.",
            "하지만 중국이 제시한 협상안을 검토해보니, 일부 조건은 받아들일 만합니다. 관세를 단계적으로 조정하면서 협상을 진행하는 것을 고려해볼 수 있겠네요."
        ]
        
        xi_dialogues = [
            "무역 통계만으로 판단하는 것은 적절하지 않습니다. 중국은 미국 기업들에게 거대한 시장을 제공하고 있으며, 양국 모두 이익을 얻고 있습니다. 대화로 해결합시다.",
            "지적재산권 보호를 위한 새로운 법안을 도입했고, 기술 협력도 공정하게 진행하고 있습니다. 상호 이해와 존중이 필요한 시점입니다.",
            "트럼프 대통령의 제안을 환영합니다. 단계적 관세 조정은 좋은 출발점이 될 것입니다. 구체적인 이행 방안을 논의해보죠."
        ]
        
        logger.info("\n=== 대화 시뮬레이션 시작 ===")
        
        # 테스트용 대화 및 평가 생성
        for turn in range(context["total_turns"]):
            current_turn = turn + 1
            logger.info(f"\n--- 턴 {current_turn} ---")
            
            # 트럼프 전문가 발언
            trump_dialogue = {
                "speaker": "trump_expert",
                "content": trump_dialogues[turn],
                "turn": current_turn,
                "timestamp": int(time.time())
            }
            dialogue_history.append(trump_dialogue)
            blockchain_manager.record_dialogue(
                speaker=trump_dialogue["speaker"],
                content=trump_dialogue["content"],
                timestamp=trump_dialogue["timestamp"]
            )
            time.sleep(1)
            
            # 평가 에이전트들의 평가 수행
            for agent_name, evaluator in context["evaluation_agents"].items():
                evaluation = evaluator.evaluate(
                    dialogue_history=[d["content"] for d in dialogue_history],
                    current_turn=trump_dialogue,
                    shared_knowledge=shared_knowledge
                )
                
                blockchain_manager.record_evaluation(
                    evaluator=agent_name,
                    subject="trump_expert",
                    evaluation_type=evaluator.type,
                    score=evaluation["score"],
                    reasoning=evaluation["reasoning"],
                    turn_number=current_turn
                )
            
            time.sleep(1)
            
            # 시진핑 전문가 발언
            xi_dialogue = {
                "speaker": "xi_expert",
                "content": xi_dialogues[turn],
                "turn": current_turn,
                "timestamp": int(time.time())
            }
            dialogue_history.append(xi_dialogue)
            blockchain_manager.record_dialogue(
                speaker=xi_dialogue["speaker"],
                content=xi_dialogue["content"],
                timestamp=xi_dialogue["timestamp"]
            )
            time.sleep(1)
            
            # 평가 에이전트들의 평가 수행
            for agent_name, evaluator in context["evaluation_agents"].items():
                evaluation = evaluator.evaluate(
                    dialogue_history=[d["content"] for d in dialogue_history],
                    current_turn=xi_dialogue,
                    shared_knowledge=shared_knowledge
                )
                
                blockchain_manager.record_evaluation(
                    evaluator=agent_name,
                    subject="xi_expert",
                    evaluation_type=evaluator.type,
                    score=evaluation["score"],
                    reasoning=evaluation["reasoning"],
                    turn_number=current_turn
                )
            
            time.sleep(1)
            
            # 합의 프로세스 기록 (마지막 턴에서)
            if turn == context["total_turns"] - 1:
                # 각 에이전트의 합의 참여 기록
                for agent_id in context["dialogue_agents"]:
                    # 합의 투표 기록
                    blockchain_manager.blockchain.add_consensus_record({
                        "voter": agent_id,
                        "vote": True,
                        "reason": "최종 합의안에 동의",
                        "timestamp": int(time.time()),
                        "turn_number": current_turn
                    })
                    
                # 합의 달성 보상
                blockchain_manager.reward_consensus(
                    participants=context["dialogue_agents"]
                )
        
        # 결과 출력 및 검증
        logger.info("\n=== 대화 결과 분석 ===")
        
        # 전체 대화 내용 출력
        dialogue_history = blockchain_manager.get_dialogue_history()
        logger.info("\n전체 대화 내용 및 평가:")
        for entry in dialogue_history:
            logger.info(f"\n발언자: {entry['speaker']}")
            logger.info(f"턴 {entry['turn_number']}:")
            logger.info(f"내용: {entry['content']}")
            if 'quality_evaluation' in entry:
                logger.info("\n평가 결과:")
                for eval in entry['quality_evaluation']['evaluations']:
                    logger.info(f"- 평가자: {eval['evaluator']}")
                    logger.info(f"  평가 유형: {eval['evaluation_type']}")
                    logger.info(f"  점수: {eval['score']}")
                    logger.info(f"  평가 의견: {eval['reasoning']}")
        
        print_detailed_results(blockchain_manager, context)
        
        # 검증 실행
        verifications = [
            (verify_dialogue_balance(blockchain_manager), "대화 참여가 불균형합니다"),
            (verify_token_distribution(blockchain_manager), "토큰 분배에 문제가 있습니다"),
            (verify_evaluation_impact(blockchain_manager), "평가 영향이 제대로 반영되지 않았습니다"),
            (verify_consensus_process(blockchain_manager), "합의 프로세스에 문제가 있습니다"),
            (verify_mining_rewards(blockchain_manager), "채굴 보상에 문제가 있습니다")
        ]
        
        for verification, error_message in verifications:
            assert verification, error_message
        
        logger.info("\n모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        logger.error(f"테스트 실패: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_dialogue_flow()
    if not success:
        logger.error("\n테스트가 실패했습니다.") 