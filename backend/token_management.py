from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class TokenTransaction:
    from_address: str
    to_address: str
    amount: int
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    transaction_type: str = "transfer"  # transfer, reward, penalty
    reason: Optional[str] = None

class TokenManager:
    def __init__(self):
        self.balances: Dict[str, int] = {}
        self.transactions: List[TokenTransaction] = []
        self.initial_distribution = 1000  # 초기 토큰 수량
        
    def create_account(self, address: str) -> int:
        """새로운 계정 생성 및 초기 토큰 지급"""
        if address not in self.balances:
            self.balances[address] = self.initial_distribution
            self._record_transaction(
                TokenTransaction(
                    from_address="system",
                    to_address=address,
                    amount=self.initial_distribution,
                    transaction_type="initial",
                    reason="Initial token distribution"
                )
            )
        return self.balances[address]
    
    def get_balance(self, address: str) -> int:
        """계정 잔액 조회"""
        return self.balances.get(address, 0)
    
    def transfer(self, from_address: str, to_address: str, amount: int, reason: Optional[str] = None) -> bool:
        """토큰 전송"""
        if (from_address not in self.balances or 
            self.balances[from_address] < amount):
            return False
            
        if to_address not in self.balances:
            self.create_account(to_address)
            
        self.balances[from_address] -= amount
        self.balances[to_address] += amount
        
        self._record_transaction(
            TokenTransaction(
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                reason=reason
            )
        )
        return True
    
    def reward(self, address: str, amount: int, reason: str) -> bool:
        """보상 토큰 지급"""
        if address not in self.balances:
            self.create_account(address)
            
        self.balances[address] += amount
        self._record_transaction(
            TokenTransaction(
                from_address="system",
                to_address=address,
                amount=amount,
                transaction_type="reward",
                reason=reason
            )
        )
        return True
    
    def penalize(self, address: str, amount: int, reason: str) -> bool:
        """페널티 토큰 차감"""
        if address not in self.balances or self.balances[address] < amount:
            return False
            
        self.balances[address] -= amount
        self._record_transaction(
            TokenTransaction(
                from_address=address,
                to_address="system",
                amount=amount,
                transaction_type="penalty",
                reason=reason
            )
        )
        return True
    
    def _record_transaction(self, transaction: TokenTransaction):
        """트랜잭션 기록"""
        self.transactions.append(transaction)
    
    def get_transaction_history(self, address: Optional[str] = None) -> List[TokenTransaction]:
        """트랜잭션 기록 조회"""
        if address is None:
            return self.transactions
        return [tx for tx in self.transactions 
                if tx.from_address == address or tx.to_address == address]
    
    def calculate_rewards(self, participation_data: Dict[str, Dict]) -> Dict[str, int]:
        """참여도 기반 보상 계산"""
        rewards = {}
        for address, data in participation_data.items():
            # 기본 점수 계산
            base_score = data.get('messages', 0) * 10  # 메시지당 10 토큰
            
            # 평가 점수 반영
            evaluation_score = data.get('evaluation_score', 0) * 100  # 평가 점수 * 100
            
            # 합의 참여 보너스
            consensus_bonus = data.get('consensus_participation', 0) * 50  # 합의 참여당 50 토큰
            
            # 총 보상 계산
            total_reward = int(base_score + evaluation_score + consensus_bonus)
            rewards[address] = total_reward
            
            # 보상 지급
            self.reward(address, total_reward, "Participation reward")
            
        return rewards

def test_token_manager():
    """토큰 관리 시스템 테스트"""
    manager = TokenManager()
    
    # 계정 생성 테스트
    address1 = "user1"
    address2 = "user2"
    
    balance1 = manager.create_account(address1)
    assert balance1 == 1000
    assert manager.get_balance(address1) == 1000
    
    # 전송 테스트
    success = manager.transfer(address1, address2, 500, "Test transfer")
    assert success
    assert manager.get_balance(address1) == 500
    assert manager.get_balance(address2) == 1500
    
    # 보상 테스트
    manager.reward(address1, 200, "Participation reward")
    assert manager.get_balance(address1) == 700
    
    # 페널티 테스트
    manager.penalize(address2, 300, "Violation penalty")
    assert manager.get_balance(address2) == 1200
    
    # 참여도 기반 보상 테스트
    participation_data = {
        address1: {
            'messages': 5,
            'evaluation_score': 0.8,
            'consensus_participation': 2
        },
        address2: {
            'messages': 3,
            'evaluation_score': 0.9,
            'consensus_participation': 1
        }
    }
    
    rewards = manager.calculate_rewards(participation_data)
    assert rewards[address1] > 0
    assert rewards[address2] > 0
    
    # 트랜잭션 히스토리 테스트
    history = manager.get_transaction_history(address1)
    assert len(history) > 0
    
    print("모든 토큰 관리 시스템 테스트 통과!")
    return manager

if __name__ == "__main__":
    test_token_manager() 