import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from langchain.tools import Tool
from langchain.agents import tool

from backend.agents import ReactAgent, DialogueAgent
from backend.graph import DialogueGraph
from backend.blockchain import BlockchainManager
from backend.report import ReportGenerator
from backend.knowledge import KnowledgeBase
from backend.logger import setup_logger

# FastAPI 앱 초기화
app = FastAPI(title="ConsenChain API", version="1.0.0")
logger = setup_logger(__name__)

# Pydantic 모델
class DialogueRequest(BaseModel):
    message: str
    agent_id: str
    context: Optional[List[Dict]] = None

class DialogueResponse(BaseModel):
    response: str
    evaluation: Optional[Dict] = None
    consensus_reached: Optional[bool] = None
    turn_count: int

class SimulationConfig(BaseModel):
    max_turns: int = 20
    consensus_threshold: float = 0.75
    evaluation_interval: int = 5
    agents: List[Dict]

def load_env():
    """환경 변수 로드"""
    load_dotenv()
    required_vars = ['OPENAI_API_KEY', 'BLOCKCHAIN_RPC_URL', 'CONTRACT_ADDRESS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

def create_tools(react_agent: ReactAgent, blockchain_manager: BlockchainManager) -> Dict[str, Tool]:
    """도구 생성"""
    tools = {
        "generate_response": Tool(
            name="generate_response",
            func=react_agent.generate_response,
            description="Generate a response in the dialogue"
        ),
        "evaluate_dialogue": Tool(
            name="evaluate_dialogue",
            func=react_agent.evaluate_dialogue,
            description="Evaluate the current dialogue"
        ),
        "decide_vote": Tool(
            name="decide_vote",
            func=react_agent.decide_vote,
            description="Make a voting decision"
        ),
        "record_dialogue": Tool(
            name="record_dialogue",
            func=blockchain_manager.record_dialogue,
            description="Record dialogue to blockchain"
        ),
        "record_evaluation": Tool(
            name="record_evaluation",
            func=blockchain_manager.record_evaluation,
            description="Record evaluation to blockchain"
        ),
        "record_consensus_vote": Tool(
            name="record_consensus_vote",
            func=blockchain_manager.record_consensus_vote,
            description="Record consensus vote to blockchain"
        )
    }
    return tools

# 전역 객체 초기화
load_env()
knowledge_base = KnowledgeBase()
react_agent = ReactAgent()
dialogue_agent = DialogueAgent(knowledge_base)
blockchain_manager = BlockchainManager({
    'rpc_url': os.getenv('BLOCKCHAIN_RPC_URL'),
    'contract_address': os.getenv('CONTRACT_ADDRESS'),
    'private_key': os.getenv('PRIVATE_KEY')
})
tools = create_tools(react_agent, blockchain_manager)
dialogue_graph = DialogueGraph(tools)

@app.post("/dialogue", response_model=DialogueResponse)
async def process_dialogue(request: DialogueRequest):
    """단일 대화 처리 엔드포인트"""
    try:
        response = dialogue_agent.process_message(
            message=request.message,
            agent_id=request.agent_id,
            context=request.context
        )
        return DialogueResponse(
            response=response["message"],
            evaluation=response.get("evaluation"),
            consensus_reached=response.get("consensus_reached"),
            turn_count=response["turn_count"]
        )
    except Exception as e:
        logger.error(f"Error processing dialogue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate_dialogue(config: SimulationConfig):
    """전체 대화 시뮬레이션 실행 엔드포인트"""
    try:
        simulation_config = {
            'max_turns': config.max_turns,
            'consensus_threshold': config.consensus_threshold,
            'evaluation_interval': config.evaluation_interval,
            'blockchain_manager': blockchain_manager,
            'agents': config.agents
        }
        
        final_state = dialogue_graph.invoke(simulation_config)
        
        report_generator = ReportGenerator()
        report_file = report_generator.generate_report(
            final_state.messages,
            final_state.evaluations
        )
        
        return {
            "status": "success",
            "report_file": report_file,
            "final_state": {
                "messages": final_state.messages,
                "consensus_reached": final_state.consensus_reached,
                "turn_count": final_state.turn_count
            }
        }
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge")
async def add_knowledge(agent_id: str, documents: List[str]):
    """지식 베이스 추가 엔드포인트"""
    try:
        knowledge_base.create_agent_knowledge_base(agent_id, documents)
        return {"status": "success", "message": f"Knowledge base created for agent {agent_id}"}
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG_MODE", "True").lower() == "true"
    
    uvicorn.run(app, host=host, port=port, debug=debug)
