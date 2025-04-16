# 합의AI: 토큰으로 연결된 대화 (ConsenChain)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 개요

**합의AI (ConsenChain)** 프로젝트는 현대 사회에서 대화 중 갈등이 발생할 때, 원만한 합의에 도달하지 못하고 대화가 단절되는 문제를 해결하고자 하는 취지에서 출발했습니다.  
이 프로젝트는 AI와 블록체인 기술을 결합하여, 사람들이 실제와 유사한 환경에서 대화하는 조건을 그대로 재현하는 동시에,  
AI 에이전트들을 통해 갈등 상황에 대한 효과적인 대화 전략과 합의 방법론을 시뮬레이션하고 평가합니다.  
특히, 각 에이전트는 개인의 사회적 명성이나 신뢰(이를 토큰 보상으로 연결)를 획득하는 환경 속에서 합의를 도출하도록 설계되어 있습니다.

## 프로젝트 구조

```
project_root/
├── main.py          # 엔트리포인트 (백엔드 로직과 인터페이스)
├── backend/
│   ├── agents.py    # AI 에이전트 관련 로직
│   ├── search.py    # 검색 및 데이터 처리
│   ├── graph_flow.py # 대화 흐름 및 그래프 처리
│   ├── blockchain.py # 블록체인 연동 로직
│   ├── report.py    # 리포트 생성 로직
│   └── config.py    # 설정 파일
├── frontend/        # 프론트엔드 관련 파일
│   └── ...
└── contracts/      # 스마트 컨트랙트
    ├── ConsensusContract.sol
    └── migrations/
```

## 주요 기능 및 특징

### AI 에이전트 합의 시뮬레이션

- **대화 에이전트 구성**
- **실시간 평가 시스템**
- **합의 및 투표 프로세스**
- **최종 리포트 생성**

### 블록체인 및 토큰 보상 시스템

- **투명하고 신뢰성 있는 기록**
- **보상 메커니즘**
- **개인 신뢰 및 명성**

## 설치 및 실행

1. 환경 설정:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. 환경 변수 설정:
- `.env` 파일을 프로젝트 루트에 생성하고 다음 내용을 설정:
```
OPENAI_API_KEY=your_openai_api_key
BLOCKCHAIN_RPC_URL=your_blockchain_rpc_url
CONTRACT_ADDRESS=your_contract_address
PRIVATE_KEY=your_private_key
```

3. 실행
```bash
python main.py
```

## 라이센스

MIT License


