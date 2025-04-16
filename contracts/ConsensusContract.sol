// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ConsensusContract is ERC20, Ownable {
    struct Dialogue {
        address speaker;
        string message;
        uint256 timestamp;
    }
    
    struct Evaluation {
        string evaluator;
        string subject;
        uint256 score;
        string reasoning;
        uint256 timestamp;
    }
    
    struct ConsensusVote {
        address voter;
        bool agreed;
        uint256 timestamp;
    }
    
    Dialogue[] public dialogues;
    Evaluation[] public evaluations;
    ConsensusVote[] public votes;
    
    uint256 public constant REWARD_POOL = 100 * 10**18;  // 100 tokens
    uint256 public constant CONSENSUS_THRESHOLD = 75;     // 75%
    
    mapping(address => uint256) public participantScores;
    
    event DialogueRecorded(address speaker, string message, uint256 timestamp);
    event EvaluationRecorded(string evaluator, string subject, uint256 score, uint256 timestamp);
    event ConsensusVoteRecorded(address voter, bool agreed, uint256 timestamp);
    event RewardsDistributed(address[] participants, uint256[] amounts);
    
    constructor() ERC20("ConsensusToken", "CST") {
        _mint(address(this), REWARD_POOL);
    }
    
    function recordDialogue(string memory message, uint256 timestamp) public {
        dialogues.push(Dialogue({
            speaker: msg.sender,
            message: message,
            timestamp: timestamp
        }));
        
        emit DialogueRecorded(msg.sender, message, timestamp);
    }
    
    function recordEvaluation(
        string memory evaluator,
        string memory subject,
        uint256 score,
        string memory reasoning,
        uint256 timestamp
    ) public {
        require(score <= 100, "Score must be between 0 and 100");
        
        evaluations.push(Evaluation({
            evaluator: evaluator,
            subject: subject,
            score: score,
            reasoning: reasoning,
            timestamp: timestamp
        }));
        
        emit EvaluationRecorded(evaluator, subject, score, timestamp);
    }
    
    function recordConsensusVote(bool agreed, uint256 timestamp) public {
        votes.push(ConsensusVote({
            voter: msg.sender,
            agreed: agreed,
            timestamp: timestamp
        }));
        
        emit ConsensusVoteRecorded(msg.sender, agreed, timestamp);
    }
    
    function distributeRewards(address[] memory participants, uint256[] memory scores) public onlyOwner {
        require(participants.length == scores.length, "Arrays must have same length");
        require(participants.length > 0, "Must have at least one participant");
        
        uint256 totalScore = 0;
        for (uint256 i = 0; i < scores.length; i++) {
            totalScore += scores[i];
        }
        
        uint256[] memory rewards = new uint256[](participants.length);
        for (uint256 i = 0; i < participants.length; i++) {
            rewards[i] = (REWARD_POOL * scores[i]) / totalScore;
            _transfer(address(this), participants[i], rewards[i]);
        }
        
        emit RewardsDistributed(participants, rewards);
    }
    
    function getDialogueCount() public view returns (uint256) {
        return dialogues.length;
    }
    
    function getEvaluationCount() public view returns (uint256) {
        return evaluations.length;
    }
    
    function getVoteCount() public view returns (uint256) {
        return votes.length;
    }
    
    function getDialogue(uint256 index) public view returns (
        address speaker,
        string memory message,
        uint256 timestamp
    ) {
        require(index < dialogues.length, "Index out of bounds");
        Dialogue memory dialogue = dialogues[index];
        return (dialogue.speaker, dialogue.message, dialogue.timestamp);
    }
    
    function getEvaluation(uint256 index) public view returns (
        string memory evaluator,
        string memory subject,
        uint256 score,
        string memory reasoning,
        uint256 timestamp
    ) {
        require(index < evaluations.length, "Index out of bounds");
        Evaluation memory evaluation = evaluations[index];
        return (
            evaluation.evaluator,
            evaluation.subject,
            evaluation.score,
            evaluation.reasoning,
            evaluation.timestamp
        );
    }
    
    function getConsensusStatus() public view returns (
        uint256 totalParticipants,
        uint256 agreedCount,
        bool thresholdReached
    ) {
        uint256 agreed = 0;
        for (uint256 i = 0; i < votes.length; i++) {
            if (votes[i].agreed) {
                agreed++;
            }
        }
        
        return (
            votes.length,
            agreed,
            (agreed * 100) / votes.length >= CONSENSUS_THRESHOLD
        );
    }
} 