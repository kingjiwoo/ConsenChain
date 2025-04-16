from typing import Dict, List, Optional, Union
from web3 import Web3
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Transaction:
    def __init__(self, from_address: str, to_address: str, amount: int, 
                 timestamp: float = None, transaction_type: str = "transfer",
                 metadata: Dict = None):
        """트랜잭션 초기화
        
        Args:
            from_address: 송신자 주소
            to_address: 수신자 주소
            amount: 전송 금액
            timestamp: 트랜잭션 생성 시간 (기본값: 현재 시간)
            transaction_type: 트랜잭션 유형 (기본값: "transfer")
            metadata: 추가 메타데이터
        """
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.timestamp = timestamp or time.time()
        self.transaction_type = transaction_type
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict:
        """트랜잭션을 딕셔너리 형태로 변환"""
        return {
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "transaction_type": self.transaction_type,
            **self.metadata  # 메타데이터를 딕셔너리에 포함
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """딕셔너리로부터 트랜잭션 객체 생성"""
        # 기본 필드들을 추출
        basic_fields = {
            "from_address": data["from_address"],
            "to_address": data["to_address"],
            "amount": data["amount"],
            "timestamp": data["timestamp"],
            "transaction_type": data.get("transaction_type", "transfer")
        }
        
        # 나머지 필드들을 메타데이터로 처리
        metadata = {k: v for k, v in data.items() 
                   if k not in basic_fields}
        
        return cls(**basic_fields, metadata=metadata)

@dataclass
class DialogueEntry:
    speaker: str
    content: str
    timestamp: int
    turn_number: int

@dataclass
class EvaluationEntry:
    evaluator: str
    subject: str
    evaluation_type: str
    score: int
    reasoning: str
    timestamp: int
    turn_number: int

@dataclass
class TokenTransaction:
    agent: str
    amount: int
    reason: str
    timestamp: int
    turn_number: int

@dataclass
class ConsensusVote:
    voter: str
    vote: bool
    reason: str
    timestamp: int
    turn_number: int
    
    def to_dict(self) -> Dict:
        """ConsensusVote를 딕셔너리로 변환"""
        return {
            "voter": self.voter,
            "vote": self.vote,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "turn_number": self.turn_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConsensusVote':
        """딕셔너리에서 ConsensusVote 객체 생성"""
        return cls(
            voter=data["voter"],
            vote=data["vote"],
            reason=data["reason"],
            timestamp=data["timestamp"],
            turn_number=data["turn_number"]
        )

class Block:
    def __init__(self, timestamp: float, transactions: List[Union[Dict, Transaction, ConsensusVote]], previous_hash: str):
        """블록 초기화
        
        Args:
            timestamp: 블록 생성 시간
            transactions: 트랜잭션 목록 (딕셔너리 또는 객체)
            previous_hash: 이전 블록의 해시
        """
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = 0
        
        # 트랜잭션 처리
        self.transactions = []
        for tx in transactions:
            if isinstance(tx, dict):
                if "voter" in tx:  # ConsensusVote 데이터
                    self.transactions.append(ConsensusVote(
                        voter=tx["voter"],
                        vote=tx["vote"],
                        reason=tx["reason"],
                        timestamp=tx["timestamp"],
                        turn_number=tx["turn_number"]
                    ))
                else:  # 일반 Transaction 데이터
                    self.transactions.append(Transaction(
                        from_address=tx["from_address"],
                        to_address=tx["to_address"],
                        amount=tx["amount"],
                        timestamp=tx.get("timestamp", time.time()),
                        transaction_type=tx.get("transaction_type", "transfer"),
                        metadata=tx.get("metadata", {})
                    ))
            else:  # 이미 객체인 경우
                self.transactions.append(tx)
        
        # 블록 해시 계산
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """블록의 해시값 계산"""
        # 트랜잭션을 딕셔너리로 변환
        transaction_dicts = []
        for tx in self.transactions:
            if isinstance(tx, dict):
                transaction_dicts.append(tx)
            else:
                transaction_dicts.append(tx.to_dict())
        
        block_data = {
            'timestamp': self.timestamp,
            'transactions': transaction_dicts,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        
        # 딕셔너리를 정렬된 JSON 문자열로 변환
        data_string = json.dumps(block_data, sort_keys=True)
        
        # SHA-256 해시 계산
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int):
        """주어진 난이도에 맞춰 블록 채굴
        
        Args:
            difficulty: 채굴 난이도 (앞자리 0의 개수)
        """
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

        logger.info(f"Block mined: {self.hash}")
        
    def to_dict(self) -> Dict:
        """블록을 딕셔너리 형태로 변환"""
        return {
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        """딕셔너리로부터 블록 객체 생성"""
        transactions = [Transaction.from_dict(tx_data) for tx_data in data["transactions"]]
        block = cls(
            timestamp=data["timestamp"],
            transactions=transactions,
            previous_hash=data["previous_hash"]
        )
        block.nonce = data["nonce"]
        block.hash = data["hash"]
        return block

class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain = []
        self.difficulty = difficulty
        self.current_block = None
        self.max_transactions = 10  # 블록당 최대 트랜잭션 수
        self.pending_transactions = []  # 대기 중인 트랜잭션
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """제네시스 블록 생성"""
        genesis_block = Block(
            timestamp=time.time(),
            transactions=[],
            previous_hash="0"
        )
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        self.current_block = None
    
    def get_latest_block(self) -> Block:
        """최신 블록 반환"""
        return self.chain[-1]
    
    def add_data_to_pending(self, data: Dict):
        """대기 중인 트랜잭션에 데이터 추가"""
        # 데이터 타입에 따라 적절한 객체 생성
        if isinstance(data, dict):
            if "voter" in data:  # ConsensusVote 데이터
                transaction = ConsensusVote(
                    voter=data["voter"],
                    vote=data["vote"],
                    reason=data["reason"],
                    timestamp=data["timestamp"],
                    turn_number=data["turn_number"]
                )
            else:  # 일반 Transaction 데이터
                transaction = Transaction(
                    from_address=data["from_address"],
                    to_address=data["to_address"],
                    amount=data["amount"],
                    timestamp=data.get("timestamp", time.time()),
                    transaction_type=data.get("transaction_type", "transfer"),
                    metadata=data.get("metadata", {})
                )
        else:  # 이미 객체인 경우
            transaction = data
        
        self.pending_transactions.append(transaction)
        
        # 대기 중인 트랜잭션이 max_transactions에 도달하면 새 블록 생성
        if len(self.pending_transactions) >= self.max_transactions:
            self.create_block_from_pending()
            
    def create_block_from_pending(self) -> Optional[Block]:
        """대기 중인 트랜잭션으로 새 블록 생성"""
        if not self.pending_transactions:
            return None
            
        # 새 블록 생성
        new_block = Block(
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.chain[-1].hash
        )
        
        # 블록 채굴
        new_block.mine_block(self.difficulty)
        
        # 체인에 블록 추가
        self.chain.append(new_block)
        self.current_block = new_block
        
        # 대기 중인 트랜잭션 초기화
        self.pending_transactions = []
        
        return new_block
    
    def calculate_merkle_root(self, data: List[Dict]) -> str:
        """머클 루트 계산"""
        if not data:
            return hashlib.sha256(''.encode()).hexdigest()
            
        hashes = [hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() for d in data]
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            hashes = [hashlib.sha256((h1 + h2).encode()).hexdigest() 
                     for h1, h2 in zip(hashes[::2], hashes[1::2])]
        return hashes[0]
    
    def is_chain_valid(self) -> bool:
        """블록체인 유효성 검증"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # 현재 블록의 해시 검증
            if current_block.hash != current_block.calculate_hash():
                return False
                
            # 이전 블록 해시 연결 검증
            if current_block.previous_hash != previous_block.hash:
                return False
                
            # 작업 증명 검증
            if current_block.hash[:self.difficulty] != '0' * self.difficulty:
                return False
        
        return True
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """인덱스로 블록 조회"""
        return self.chain[index] if 0 <= index < len(self.chain) else None
    
    def get_block_by_hash(self, hash: str) -> Optional[Block]:
        """해시로 블록 조회"""
        for block in self.chain:
            if block.hash == hash:
                return block
        return None
    
    def get_transactions_by_address(self, address: str) -> List[Dict]:
        """특정 주소와 관련된 모든 트랜잭션 조회"""
        transactions = []
        for block in self.chain:
            for tx in block.transactions:
                # Transaction 객체인 경우
                if isinstance(tx, Transaction):
                    if tx.from_address == address or tx.to_address == address:
                        transactions.append(tx.to_dict())
                # ConsensusVote 객체인 경우
                elif isinstance(tx, ConsensusVote):
                    if tx.voter == address:
                        transactions.append(tx.to_dict())
                # 딕셔너리인 경우 (이전 데이터 호환성)
                elif isinstance(tx, dict):
                    if (tx.get('from_address') == address or 
                        tx.get('to_address') == address or 
                        tx.get('voter') == address):
                        transactions.append(tx)
        return transactions

    def add_token_transaction(self, transaction: Dict):
        """현재 블록에 토큰 거래 기록 추가"""
        current_block = self.chain[-1]
        current_block.transactions.append(Transaction(
            from_address=transaction["from_address"],
            to_address=transaction["to_address"],
            amount=transaction["amount"],
            timestamp=transaction["timestamp"],
            transaction_type=transaction["transaction_type"],
            metadata=transaction.get("metadata", {})
        ))
        current_block.hash = current_block.calculate_hash()
        
    def add_consensus_record(self, record: Dict):
        """합의 기록 추가"""
        consensus_vote = ConsensusVote(
            voter=record["voter"],
            vote=record["vote"],
            reason=record["reason"],
            timestamp=record["timestamp"],
            turn_number=record["turn_number"]
        )
        
        # 대기 중인 트랜잭션에 추가
        self.pending_transactions.append(consensus_vote)
        
        # 대기 중인 트랜잭션이 max_transactions에 도달하면 새 블록 생성
        if len(self.pending_transactions) >= self.max_transactions:
            self.create_block_from_pending()
            
    def get_token_balance(self, agent_name: str) -> int:
        """특정 에이전트의 토큰 잔액 조회"""
        balance = 0
        for block in self.chain:
            for tx in block.transactions:
                # Transaction 객체인 경우
                if isinstance(tx, Transaction):
                    if tx.from_address == agent_name:
                        balance -= tx.amount
                    if tx.to_address == agent_name:
                        balance += tx.amount
                # ConsensusVote 객체는 토큰 잔액에 영향을 주지 않음
                elif isinstance(tx, ConsensusVote):
                    continue
                # 딕셔너리인 경우 (이전 데이터 호환성)
                elif isinstance(tx, dict):
                    if tx.get('from_address') == agent_name:
                        balance -= tx.get('amount', 0)
                    if tx.get('to_address') == agent_name:
                        balance += tx.get('amount', 0)
        return balance
        
    def get_consensus_statistics(self, agent_name: str) -> Dict:
        """특정 에이전트의 합의 참여 통계 조회"""
        total_participations = 0
        successful_consensus = 0
        failed_consensus = 0
        
        for block in self.chain:
            for tx in block.transactions:
                if isinstance(tx, ConsensusVote) and tx.voter == agent_name:
                    total_participations += 1
                    if tx.vote:
                        successful_consensus += 1
                    else:
                        failed_consensus += 1
                        
        return {
            "total_participations": total_participations,
            "successful_consensus": successful_consensus,
            "failed_consensus": failed_consensus
        }

class BlockchainManager:
    def __init__(self, difficulty: int = 4, initial_agents: List[Dict] = None,
                 token_rewards: Dict = None, config: Dict = None):
        """블록체인 매니저 초기화
        
        Args:
            difficulty: 블록체인 난이도 (기본값: 4)
            initial_agents: 초기 에이전트 목록 (기본값: None)
            token_rewards: 토큰 보상 설정 (기본값: None)
            config: 설정 딕셔너리 (선택사항) - 다른 매개변수보다 우선순위가 높음
        """
        if config is not None:
            # config 딕셔너리가 제공된 경우 해당 값을 사용
            self.blockchain = Blockchain(config.get("difficulty", 4))
            self.token_rewards = config.get("token_rewards", {})
            initial_agents = config.get("agents", [])
        else:
            # 개별 매개변수 사용
            self.blockchain = Blockchain(difficulty)
            self.token_rewards = token_rewards or {
                "participation": 5,  # 대화 참여 보상
                "evaluation": 3,     # 평가 참여 보상
                "quality": 10,       # 높은 품질 보상
                "consensus": 5       # 합의 참여 보상
            }
        
        # 공통 초기화
        self.evaluator_history = {}  # 평가자별 평가 이력
        self.speaker_counts = {}     # 발언자별 발언 횟수
        self.consensus_rewards = {}  # 합의 보상 기록
        self.dialogue_history = []   # 대화 이력
        self.current_turn = 1        # 현재 대화 턴
        
        # 초기 토큰 분배
        initial_agents = initial_agents or []
        for agent in initial_agents:
            if isinstance(agent, dict):
                self.consensus_rewards[agent["name"]] = agent.get("initial_balance", 0)
            elif isinstance(agent, str):
                self.consensus_rewards[agent] = 0
                
    def initialize_agent(self, agent_id: str, initial_tokens: int = 0):
        """새로운 에이전트 초기화
        
        Args:
            agent_id: 에이전트 식별자
            initial_tokens: 초기 토큰 수 (기본값: 0)
        """
        if agent_id not in self.consensus_rewards:
            self.consensus_rewards[agent_id] = initial_tokens
            # 초기 토큰 지급 트랜잭션 생성
            if initial_tokens > 0:
                self.blockchain.add_data_to_pending({
                    "from_address": "system",
                    "to_address": agent_id,
                    "amount": initial_tokens,
                    "transaction_type": "initialization",
                    "metadata": {
                        "reason": "Initial token allocation"
                    }
                })
                # 블록 생성
                self.blockchain.create_block_from_pending()
        logger.info(f"Agent {agent_id} initialized with {initial_tokens} tokens")

    def record_dialogue(self, speaker: str, content: str, timestamp: int):
        """대화 내용 기록
        
        Args:
            speaker: 발언자
            content: 대화 내용
            timestamp: 타임스탬프
        """
        # 대화 내용 로깅
        logger.info(f"\n=== 새로운 대화 기록 ===")
        logger.info(f"발언자: {speaker}")
        logger.info(f"턴 번호: {self.current_turn}")
        logger.info(f"내용: {content}")
        logger.info(f"시간: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        
        dialogue_entry = DialogueEntry(
            speaker=speaker,
            content=content,
            timestamp=timestamp,
            turn_number=self.current_turn
        )
        
        # 대화 이력에 추가
        self.dialogue_history.append(dialogue_entry)
        
        # 발언 횟수 업데이트
        self.update_speaker_count(speaker)
        
        # 대화 참여 보상 지급
        participation_reward = self.token_rewards.get("participation", 5)
        
        # 블록체인에 트랜잭션 추가
        transaction_data = {
            "from_address": "system",
            "to_address": speaker,
            "amount": participation_reward,
            "timestamp": timestamp,
            "transaction_type": "dialogue",
            "metadata": {
                "content": content,
                "turn_number": self.current_turn,
                "reward_type": "participation"
            }
        }
        self.blockchain.add_data_to_pending(transaction_data)
        
        # 일정 수의 트랜잭션이 쌓이면 블록 생성
        if len(self.blockchain.pending_transactions) >= 5:
            self.blockchain.create_block_from_pending()
            
        logger.info(f"대화 참여 보상: {participation_reward} 토큰")
        logger.info("=== 대화 기록 완료 ===\n")
    
    def get_dialogue_history(self) -> List[Dict]:
        """전체 대화 이력 조회"""
        history = []
        for entry in self.dialogue_history:
            dialogue_dict = asdict(entry)
            
            # 해당 대화에 대한 평가 찾기
            evaluations = [e for e in self.get_evaluation_history() 
                         if isinstance(e, dict) and
                         e.get("turn_number") == entry.turn_number and 
                         e.get("subject") == entry.speaker]
            
            if evaluations:
                # 평가 점수 평균 계산
                total_score = sum(e.get("score", 0) for e in evaluations)
                avg_score = total_score / len(evaluations)
                dialogue_dict["quality_evaluation"] = {
                    "total_score": avg_score,
                    "evaluations": evaluations
                }
            
            # 대화 내용 로깅
            logger.info(f"\n대화 기록 - 턴 {entry.turn_number}:")
            logger.info(f"발언자: {entry.speaker}")
            logger.info(f"내용: {entry.content}")
            if "quality_evaluation" in dialogue_dict:
                logger.info(f"평가 점수: {dialogue_dict['quality_evaluation']['total_score']:.2f}")
                for eval in evaluations:
                    logger.info(f"- 평가자: {eval.get('evaluator')}")
                    logger.info(f"  점수: {eval.get('score')}")
                    logger.info(f"  평가: {eval.get('reasoning')}")
            
            history.append(dialogue_dict)
        
        return history
    
    def get_evaluation_history(self) -> List[Dict]:
        """전체 평가 이력 조회"""
        all_evaluations = []
        for evaluator, evals in self.evaluator_history.items():
            all_evaluations.extend(evals)
        return all_evaluations
    
    def get_agent_status(self, agent: str) -> Dict:
        """에이전트의 현재 상태 조회"""
        transactions = self.blockchain.get_transactions_by_address(agent)
        
        # 평가 참여 횟수 계산
        evaluation_count = sum(1 for tx in transactions 
                             if isinstance(tx, dict) and
                             tx.get("transaction_type") == "evaluation" and 
                             tx.get("metadata", {}).get("evaluator") == agent)
        
        # 받은 평가 점수 계산
        received_evaluations = [tx for tx in transactions 
                              if isinstance(tx, dict) and
                              tx.get("transaction_type") == "evaluation" and 
                              tx.get("to_address") == agent]
        
        avg_score = 0.0
        if received_evaluations:
            avg_score = sum(tx.get("amount", 0) for tx in received_evaluations) / len(received_evaluations)
        
        # 토큰 거래 이력 가공
        token_history = []
        for tx in transactions:
            if not isinstance(tx, dict):
                continue
                
            tx_type = tx.get("transaction_type", "")
            if tx_type in ["dialogue", "evaluation", "consensus_vote", "evaluation_reward", "quality_reward"]:
                history_entry = {
                    "amount": tx.get("amount", 0),
                    "reason": self._get_transaction_reason(tx)
                }
                token_history.append(history_entry)
        
            return {
            "token_balance": self.blockchain.get_token_balance(agent),
            "dialogue_count": self.speaker_counts.get(agent, 0),
            "evaluation_count": evaluation_count,
            "average_evaluation_score": avg_score,
            "token_history": token_history
        }
    
    def _get_transaction_reason(self, transaction: Dict) -> str:
        """트랜잭션의 이유 문자열 생성"""
        if not isinstance(transaction, dict):
            return "알 수 없는 거래"
            
        tx_type = transaction.get("transaction_type", "")
        metadata = transaction.get("metadata", {})
        
        if tx_type == "dialogue":
            return "대화 참여 보상"
        elif tx_type == "evaluation":
            if metadata.get("evaluator"):
                eval_type = metadata.get("evaluation_type", "일반 평가")
                return f"평가 수행: {eval_type}"
            else:
                amount = transaction.get("amount", 0)
                return f"평가 점수 획득: {amount}점"
        elif tx_type == "consensus_vote":
            result = "성공" if metadata.get("vote") else "실패"
            return f"합의 투표 참여 ({result})"
        elif tx_type == "evaluation_reward":
            return metadata.get("reason", "평가 참여 보상")
        elif tx_type == "quality_reward":
            return "높은 품질의 기여 보상"
        return "기타 거래"

    def next_turn(self):
        """다음 대화 턴으로 진행"""
        self.current_turn += 1
        logger.info(f"턴 {self.current_turn} 시작")

    def get_evaluator_history(self, evaluator: str = None) -> List[Dict]:
        """평가자별 평가 이력 반환
        
        Args:
            evaluator: 특정 평가자의 이력만 조회하고 싶을 때 지정 (선택사항)
        
        Returns:
            List[Dict]: 평가 이력 목록. 각 항목은 평가자, 대상, 점수, 평가 내용 등을 포함
        """
        if evaluator is not None:
            evaluations = self.evaluator_history.get(evaluator, [])
            # 각 평가 항목에 평가자 정보 추가
            for eval in evaluations:
                if isinstance(eval, dict):
                    eval['evaluator'] = evaluator
            return evaluations
        
        # 전체 평가 이력 조회 시
        all_evaluations = []
        for eval_name, evals in self.evaluator_history.items():
            for eval in evals:
                if isinstance(eval, dict):
                    eval_copy = eval.copy()
                    eval_copy['evaluator'] = eval_name
                    all_evaluations.append(eval_copy)
        
        return sorted(all_evaluations, key=lambda x: x.get('timestamp', 0))

    def update_evaluator_history(self, evaluator: str, evaluation: Dict):
        """평가자의 평가 이력 업데이트"""
        if evaluator not in self.evaluator_history:
            self.evaluator_history[evaluator] = []
        
        # 평가 데이터가 딕셔너리인지 확인하고 필요한 필드 추가
        if isinstance(evaluation, dict):
            eval_copy = evaluation.copy()
            if 'timestamp' not in eval_copy:
                eval_copy['timestamp'] = int(time.time())
            self.evaluator_history[evaluator].append(eval_copy)
        else:
            logger.warning(f"Invalid evaluation data format for evaluator {evaluator}")

    def get_speaker_counts(self) -> Dict[str, int]:
        """발언자별 발언 횟수 반환"""
        return self.speaker_counts
        
    def get_consensus_rewards(self, participant: str) -> int:
        """참여자의 누적 합의 보상 조회"""
        return self.consensus_rewards.get(participant, 0)
        
    def reward_consensus(self, participants: List[str], reward_amount: int = 10):
        """합의 참여자들에게 보상 지급
        
        Args:
            participants: 합의 참여자 목록
            reward_amount: 지급할 보상 금액 (기본값: 10)
        """
        try:
            current_turn = self.current_turn
            timestamp = int(time.time())
            
            # 각 참여자에 대한 보상 처리
            for participant in participants:
                # 합의 참여 기록 확인
                consensus_stats = self.blockchain.get_consensus_statistics(participant)
                if consensus_stats['total_participations'] > 0:
                    # 보상 트랜잭션 생성
                    reward_tx = Transaction(
                        from_address="system",
                        to_address=participant,
                        amount=reward_amount,
                        timestamp=timestamp,
                        transaction_type="consensus_reward",
                        metadata={
                            "reason": "합의 참여 보상",
                            "turn_number": current_turn,
                            "consensus_stats": consensus_stats
                        }
                    )
                    
                    # 트랜잭션을 블록체인에 추가
                    self.blockchain.add_data_to_pending(reward_tx)
                    
                    # 합의 보상 기록 업데이트
                    current_rewards = self.consensus_rewards.get(participant, 0)
                    self.consensus_rewards[participant] = current_rewards + reward_amount
                    
                    logger.info(f"합의 보상 지급: {participant}에게 {reward_amount} 토큰")
            
            # 보상 트랜잭션들로 새 블록 생성
            if self.blockchain.pending_transactions:
                new_block = self.blockchain.create_block_from_pending()
                if new_block:
                    logger.info(f"합의 보상 블록 생성: {new_block.hash}")
                
        except Exception as e:
            logger.error(f"합의 보상 처리 중 오류 발생: {str(e)}")
            raise

    def update_speaker_count(self, speaker: str):
        """발언자의 발언 횟수 업데이트"""
        if speaker not in self.speaker_counts:
            self.speaker_counts[speaker] = 0
        self.speaker_counts[speaker] += 1
        
    def get_speaker_statistics(self, speaker: str) -> Dict:
        """발언자의 통계 정보 반환"""
        return {
            "total_speeches": self.speaker_counts.get(speaker, 0),
            "total_rewards": self.get_consensus_rewards(speaker),
            "evaluations_received": len([
                eval for evals in self.evaluator_history.values()
                for eval in evals
                if eval.get("subject") == speaker
            ])
        }
        
    def get_all_statistics(self) -> Dict:
        """전체 통계 정보 반환"""
        return {
            "speaker_counts": self.get_speaker_counts(),
            "consensus_rewards": self.consensus_rewards,
            "evaluator_history": self.get_evaluator_history()
        }

    def record_evaluation(self, evaluator: str, subject: str, evaluation_type: str,
                         score: int, reasoning: str, turn_number: int):
        """평가 내용 기록
        
        Args:
            evaluator: 평가자
            subject: 평가 대상
            evaluation_type: 평가 유형
            score: 평가 점수
            reasoning: 평가 근거
            turn_number: 평가 대상 턴 번호
        """
        current_time = int(time.time())
        
        # 평가 데이터 생성
        evaluation_data = {
            "evaluator": evaluator,
            "subject": subject,
            "evaluation_type": evaluation_type,
            "score": score,
            "reasoning": reasoning,
            "timestamp": current_time,
            "turn_number": turn_number
        }
        
        # 평가 이력 업데이트
        self.update_evaluator_history(evaluator, evaluation_data)
        
        # 평가 점수 트랜잭션 추가
        score_tx_data = {
            "from_address": "system",
            "to_address": subject,
            "amount": score,
            "timestamp": current_time,
            "transaction_type": "evaluation",
            "metadata": {
                "evaluator": evaluator,
                "evaluation_type": evaluation_type,
                "reasoning": reasoning,
                "turn_number": turn_number,
                "reward_type": "evaluation_score"
            }
        }
        self.blockchain.add_data_to_pending(score_tx_data)
        
        # 평가자에게 평가 참여 보상 지급
        evaluation_reward = self.token_rewards.get("evaluation", 3)
        reward_tx_data = {
            "from_address": "system",
            "to_address": evaluator,
            "amount": evaluation_reward,
            "timestamp": current_time,
            "transaction_type": "evaluation_reward",
            "metadata": {
                "reason": "평가 참여 보상",
                "evaluation_type": evaluation_type,
                "subject": subject,
                "turn_number": turn_number
            }
        }
        self.blockchain.add_data_to_pending(reward_tx_data)
        
        # 품질 보상 지급 (높은 점수의 경우)
        if score >= 8:
            quality_reward = self.token_rewards.get("quality", 10)
            quality_tx_data = {
                "from_address": "system",
                "to_address": subject,
                "amount": quality_reward,
                "timestamp": current_time,
                "transaction_type": "quality_reward",
                "metadata": {
                    "reason": "높은 품질의 기여",
                    "evaluation_type": evaluation_type,
                    "score": score,
                    "evaluator": evaluator
                }
            }
            self.blockchain.add_data_to_pending(quality_tx_data)
        
        # 일정 수의 트랜잭션이 쌓이면 블록 생성
        if len(self.blockchain.pending_transactions) >= 5:
            self.blockchain.create_block_from_pending()
            
        # 평가 기록 로깅
        logger.info(f"\n평가 기록:")
        logger.info(f"평가자: {evaluator}")
        logger.info(f"대상: {subject}")
        logger.info(f"턴 번호: {turn_number}")
        logger.info(f"평가 유형: {evaluation_type}")
        logger.info(f"점수: {score}")
        logger.info(f"평가 내용: {reasoning}")
        if score >= 8:
            logger.info(f"품질 보상 지급: {quality_reward} 토큰")

    def verify_dialogue_balance(self) -> bool:
        """대화 참여의 균형을 검증"""
        # 발언 횟수가 가장 많은 참여자와 가장 적은 참여자의 차이가 2회 이하여야 함
        if not self.speaker_counts:
            return True
            
        counts = list(self.speaker_counts.values())
        return max(counts) - min(counts) <= 2
        
    def verify_token_distribution(self) -> bool:
        """토큰 분배의 적절성을 검증"""
        try:
            # 모든 참여자의 현재 토큰 잔액 확인
            balances = {}
            for agent in self.speaker_counts.keys():
                balance = self.blockchain.get_token_balance(agent)
                if balance >= 0:  # 음수 잔액은 제외
                    balances[agent] = balance
            
            if not balances:
                logger.warning("검증 가능한 토큰 잔액이 없습니다.")
                return False
            
            # 토큰 분배 통계 계산
            total_tokens = sum(balances.values())
            avg_balance = total_tokens / len(balances)
            max_balance = max(balances.values())
            min_balance = min(balances.values())
            
            # 결과 로깅
            logger.info("\n=== 토큰 분배 검증 ===")
            logger.info(f"총 토큰량: {total_tokens}")
            logger.info(f"평균 보유량: {avg_balance:.2f}")
            logger.info(f"최대 보유량: {max_balance}")
            logger.info(f"최소 보유량: {min_balance}")
            logger.info("\n개별 보유량:")
            for agent, balance in balances.items():
                logger.info(f"{agent}: {balance} 토큰")
            
            # 검증 기준:
            # 1. 모든 참여자가 최소 초기 토큰량의 50% 이상 보유
            initial_token_threshold = 50  # 초기 토큰의 50%
            if min_balance < initial_token_threshold:
                logger.warning(f"일부 참여자의 토큰량이 너무 적습니다. (최소: {min_balance} < 기준: {initial_token_threshold})")
                return False
            
            # 2. 최대-최소 차이가 평균의 150%를 넘지 않음
            max_difference_allowed = avg_balance * 1.5
            actual_difference = max_balance - min_balance
            
            if actual_difference > max_difference_allowed:
                logger.warning("토큰 분배가 불균형합니다.")
                logger.warning(f"현재 차이: {actual_difference}")
                logger.warning(f"허용 차이: {max_difference_allowed}")
                return False
            
            # 3. 모든 참여자가 양수의 토큰을 보유
            if min_balance <= 0:
                logger.warning("일부 참여자의 토큰량이 0 이하입니다.")
                return False
            
            logger.info("\n토큰 분배가 적절합니다.")
            return True
            
        except Exception as e:
            logger.error(f"토큰 분배 검증 중 오류 발생: {str(e)}")
            return False
        
    def verify_evaluation_impact(self) -> bool:
        """평가가 토큰 분배에 미치는 영향을 검증"""
        evaluation_rewards = {}
        
        # 각 참여자별 평가 관련 보상 집계
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if not isinstance(tx, dict):
                    continue
                    
                tx_type = tx.get("transaction_type", "")
                if tx_type in ["evaluation", "evaluation_reward", "quality_reward"]:
                    to_address = tx.get("to_address")
                    if to_address:
                        if to_address not in evaluation_rewards:
                            evaluation_rewards[to_address] = 0
                        evaluation_rewards[to_address] += tx.get("amount", 0)
        
        if not evaluation_rewards:
            return True
            
        # 평가 보상이 전체 토큰의 20% 이상을 차지해야 함
        total_tokens = sum(self.blockchain.get_token_balance(agent) 
                         for agent in self.speaker_counts.keys())
        total_evaluation_rewards = sum(evaluation_rewards.values())
        
        return total_evaluation_rewards >= (total_tokens * 0.2)
        
    def get_transaction_history(self, agent: str) -> List[Dict]:
        """특정 에이전트의 전체 트랜잭션 이력 조회"""
        transactions = []
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if not isinstance(tx, dict):
                    continue
                    
                if (tx.get("from_address") == agent or 
                    tx.get("to_address") == agent):
                    tx_copy = tx.copy()
                    tx_copy["block_hash"] = block.hash
                    transactions.append(tx_copy)
        return transactions
        
    def get_evaluation_summary(self, agent: str) -> Dict:
        """특정 에이전트의 평가 관련 통계 요약"""
        evaluations_given = []
        evaluations_received = []
        
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if not isinstance(tx, dict):
                    continue
                    
                if tx.get("transaction_type") == "evaluation":
                    metadata = tx.get("metadata", {})
                    if metadata.get("evaluator") == agent:
                        evaluations_given.append(tx)
                    if tx.get("to_address") == agent:
                        evaluations_received.append(tx)
        
        return {
            "evaluations_given": len(evaluations_given),
            "evaluations_received": len(evaluations_received),
            "average_score_given": (sum(tx.get("amount", 0) for tx in evaluations_given) / 
                                  len(evaluations_given)) if evaluations_given else 0,
            "average_score_received": (sum(tx.get("amount", 0) for tx in evaluations_received) / 
                                     len(evaluations_received)) if evaluations_received else 0
        }

    def get_token_distribution_history(self) -> List[Dict]:
        """토큰 분배 이력 조회"""
        distribution_history = []
        balances = {}
        
        # 초기 토큰 분배 상태 설정
        for agent in self.speaker_counts.keys():
            balances[agent] = self.blockchain.get_token_balance(agent)
        
        # 블록체인의 모든 블록을 순회하며 토큰 분배 상태 추적
        for block in self.blockchain.chain:
            block_balances = balances.copy()
            has_changes = False
            
            for tx in block.transactions:
                # Transaction 객체인 경우
                if isinstance(tx, Transaction):
                    from_addr = tx.from_address
                    to_addr = tx.to_address
                    amount = tx.amount
                    
                    if from_addr and from_addr != "system":
                        block_balances[from_addr] = block_balances.get(from_addr, 0) - amount
                        has_changes = True
                    if to_addr:
                        block_balances[to_addr] = block_balances.get(to_addr, 0) + amount
                        has_changes = True
                
                # ConsensusVote 객체는 토큰 분배에 영향을 주지 않음
                elif isinstance(tx, ConsensusVote):
                    continue
                
                # 딕셔너리인 경우 (이전 데이터 호환성)
                elif isinstance(tx, dict):
                    from_addr = tx.get("from_address")
                    to_addr = tx.get("to_address")
                    amount = tx.get("amount", 0)
                    
                    if from_addr and from_addr != "system":
                        block_balances[from_addr] = block_balances.get(from_addr, 0) - amount
                        has_changes = True
                    if to_addr:
                        block_balances[to_addr] = block_balances.get(to_addr, 0) + amount
                        has_changes = True
            
            # 변화가 있는 경우에만 기록
            if has_changes:
                distribution_history.append({
                    "block_hash": block.hash,
                    "timestamp": block.timestamp,
                    "balances": {k: v for k, v in block_balances.items() if k != "system"}
                })
                balances = block_balances.copy()
        
        # 최소한 하나의 기록은 남기기
        if not distribution_history and balances:
            distribution_history.append({
                "block_hash": self.blockchain.get_latest_block().hash,
                "timestamp": int(time.time()),
                "balances": {k: v for k, v in balances.items() if k != "system"}
            })
        
        return distribution_history

def test_blockchain():
    """블록체인 기능 테스트"""
    # 블록체인 매니저 초기화
    config = {
        "initial_tokens": 100,
        "agents": [
            {"name": "AI_Expert", "initial_balance": 100},
            {"name": "Ethics_Expert", "initial_balance": 100}
        ],
        "token_rewards": {
            "participation": 5,
            "evaluation": 3,
            "quality": 10,
            "consensus": 15,
            "constructive": 8,
            "knowledge": 7,
            "detailed": 5,
            "guidance": 7,
            "vote": 4,
            "success": 20,
            "mediator": 10
        }
    }
    manager = BlockchainManager(config)
    
    print("\n=== 블록체인 테스트 시작 ===")
    
    # 1. 대화 기록 테스트
    print("\n1. 대화 기록 테스트")
    manager.record_dialogue(
        speaker="AI_Expert",
        content="AI 윤리 가이드라인에 대해 논의해보겠습니다. 연구 결과에 따르면, "
                "데이터 프라이버시와 알고리즘 투명성이 가장 중요한 요소입니다.",
        timestamp=int(time.time())
    )
    
    manager.record_dialogue(
        speaker="Ethics_Expert",
        content="동의합니다. 구체적인 사례를 통해 분석해보면, 프라이버시 보호와 "
                "투명성 확보는 상호보완적인 관계를 가집니다.",
        timestamp=int(time.time())
    )
    
    # 2. 평가 기록 테스트
    print("\n2. 평가 기록 테스트")
    manager.record_evaluation(
        evaluator="Ethics_Expert",
        subject="AI_Expert",
        evaluation_type="dialogue_quality",
        score=9,
        reasoning="논리적이고 구체적인 근거를 제시하며, 건설적인 토론을 이끌어내는 발언이었습니다. "
                 "특히 연구 결과를 인용하여 주장의 신뢰성을 높였습니다.",
        turn_number=1
    )
    
    # 3. 합의 프로세스 테스트
    print("\n3. 합의 프로세스 테스트")
    # 합의 보상 테스트
    participants = ["AI_Expert", "Ethics_Expert"]
    manager.reward_consensus(participants)
    print(f"합의 보상 지급 완료: {participants}")
    
    # 4. 토큰 분배 확인
    print("\n4. 토큰 분배 현황")
    for agent in ["AI_Expert", "Ethics_Expert"]:
        status = manager.get_agent_status(agent)
        print(f"\n{agent} 상태:")
        print(f"- 토큰 잔액: {status['token_balance']}")
        print(f"- 대화 참여 횟수: {status['dialogue_count']}")
        print(f"- 평가 참여 횟수: {status['evaluation_count']}")
        print(f"- 평균 평가 점수: {status['average_evaluation_score']}")
        
        # 토큰 이력 출력
        print("\n토큰 거래 이력:")
        for tx in status['token_history']:
            print(f"- {tx['amount']} 토큰: {tx['reason']}")
    
    # 5. 블록체인 상태 확인
    print("\n5. 블록체인 상태")
    print(f"체인 길이: {len(manager.blockchain.chain)}")
    print(f"마지막 블록 해시: {manager.blockchain.get_latest_block().hash}")
    print(f"체인 유효성: {manager.blockchain.is_chain_valid()}")
    
    # 6. 대화 이력 확인
    print("\n6. 대화 이력")
    dialogue_history = manager.get_dialogue_history()
    for entry in dialogue_history:
        print(f"\n발언자: {entry['speaker']}")
        print(f"메시지: {entry['content']}")
        if 'quality_evaluation' in entry:
            print(f"품질 평가 점수: {entry['quality_evaluation']['total_score']:.2f}")
    
    # 7. 평가 이력 확인
    print("\n7. 평가 이력")
    evaluation_history = manager.get_evaluator_history()
    for entry in evaluation_history:
        print(f"\n평가자: {entry['evaluator']}")
        print(f"대상: {entry['subject']}")
        print(f"점수: {entry['score']}")
        print(f"평가 내용: {entry['reasoning']}")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_blockchain() 