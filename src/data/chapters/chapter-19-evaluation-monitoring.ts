import { Chapter } from '../types';

export const evaluationMonitoringChapter: Chapter = {
  id: 'evaluation-monitoring',
  title: 'Evaluation and Monitoring',
  subtitle: 'Systematic Performance Assessment and Continuous Monitoring of AI Agents',
  description: 'Implement comprehensive evaluation frameworks and monitoring systems to assess agent performance, detect anomalies, and ensure continuous alignment with operational requirements.',
  readingTime: '32 min read',
  overview: `This chapter examines methodologies that allow intelligent agents to systematically assess their performance, monitor progress toward goals, and detect operational anomalies. While Chapter 11 outlines goal setting and monitoring, and Chapter 17 addresses reasoning mechanisms, this chapter focuses on the continuous, often external, measurement of an agent's effectiveness, efficiency, and compliance with requirements.

This includes defining metrics, establishing feedback loops, and implementing reporting systems to ensure agent performance aligns with expectations in operational environments. The evaluation framework covers response accuracy assessment, latency monitoring, resource consumption tracking, and sophisticated techniques like agent trajectory analysis and LLM-as-a-Judge evaluations for nuanced quality assessment.

Advanced topics include the evolution from simple AI agents to formal "contractor" systems with explicit agreements, hierarchical task decomposition, and quality-focused iterative execution. This transformation enables reliable deployment in mission-critical domains where trust and accountability are paramount.`,
  keyPoints: [
    'Comprehensive performance tracking in live systems measuring accuracy, latency, resource consumption, and compliance with operational requirements',
    'Advanced evaluation methodologies including A/B testing for agent improvements, drift detection, and anomaly identification in agent behavior patterns',
    'Agent trajectory analysis examining decision-making sequences, tool selection strategies, and task execution efficiency against ground-truth benchmarks',
    'LLM-as-a-Judge evaluation frameworks enabling nuanced assessment of subjective qualities like helpfulness, coherence, and domain-specific expertise',
    'Multi-agent system evaluation focusing on collaborative effectiveness, inter-agent communication quality, and distributed task coordination success',
    'Token usage monitoring and cost optimization for LLM-powered agents with detailed resource allocation and billing management strategies',
    'Evolution to contractor-based agent systems with formal agreements, negotiation capabilities, and hierarchical task decomposition for complex projects',
    'Integration with evaluation platforms like Google ADK providing structured testing methodologies, evalset management, and automated assessment pipelines'
  ],
  codeExample: `# Comprehensive Agent Evaluation and Monitoring Framework
# Advanced performance assessment and continuous monitoring system

import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import statistics
from collections import defaultdict
import hashlib

# External dependencies for comprehensive evaluation
import google.generativeai as genai
from pydantic import BaseModel, Field

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvaluationMetric(Enum):
    """Types of evaluation metrics for agent assessment."""
    ACCURACY = "accuracy"
    LATENCY = "latency" 
    TOKEN_USAGE = "token_usage"
    HELPFULNESS = "helpfulness"
    TRAJECTORY_MATCH = "trajectory_match"
    COST_EFFICIENCY = "cost_efficiency"
    SAFETY_COMPLIANCE = "safety_compliance"

class TrajectoryMatchType(Enum):
    """Types of trajectory matching for agent evaluation."""
    EXACT_MATCH = "exact_match"
    IN_ORDER_MATCH = "in_order_match"
    ANY_ORDER_MATCH = "any_order_match"
    PRECISION_RECALL = "precision_recall"
    SINGLE_TOOL = "single_tool"

# --- Data Models ---
@dataclass
class AgentInteraction:
    """Complete record of an agent interaction for evaluation."""
    interaction_id: str
    timestamp: datetime
    user_input: str
    agent_output: str
    expected_output: Optional[str] = None
    trajectory: List[str] = field(default_factory=list)
    expected_trajectory: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Results from a comprehensive agent evaluation."""
    interaction_id: str
    metric_type: EvaluationMetric
    score: float
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class LLMJudgeEvaluation(BaseModel):
    """Structured evaluation result from LLM-as-a-Judge."""
    overall_score: int = Field(description="Overall quality score from 1-5")
    rationale: str = Field(description="Detailed explanation of the evaluation")
    detailed_feedback: List[str] = Field(description="Specific feedback points")
    concerns: List[str] = Field(description="Any identified concerns or issues")
    recommended_action: str = Field(description="Recommended next steps")

class ComprehensiveAgentEvaluator:
    """
    Advanced evaluation framework for comprehensive agent performance assessment
    with multiple evaluation methodologies and continuous monitoring capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.interactions: List[AgentInteraction] = []
        self.evaluation_results: List[EvaluationResult] = []
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        
        # Initialize evaluation components
        self._setup_llm_judge()
        self._setup_monitoring()
        
        self.logger = logging.getLogger("AgentEvaluator")
        self.logger.info("Comprehensive Agent Evaluator initialized")
    
    def _setup_llm_judge(self):
        """Initialize LLM-as-a-Judge evaluation system."""
        try:
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            self.judge_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.judge_enabled = True
        except Exception as e:
            self.logger.warning(f"LLM Judge unavailable: {e}")
            self.judge_enabled = False
    
    def _setup_monitoring(self):
        """Initialize monitoring and alerting systems."""
        self.monitoring_active = True
        self.alert_callbacks: List[Callable] = []
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
    
    async def record_interaction(self, 
                               user_input: str,
                               agent_output: str,
                               expected_output: str = None,
                               trajectory: List[str] = None,
                               expected_trajectory: List[str] = None,
                               metadata: Dict[str, Any] = None) -> str:
        """
        Record a complete agent interaction for comprehensive evaluation.
        
        Args:
            user_input: The user's input to the agent
            agent_output: The agent's response
            expected_output: Expected/ground truth response (if available)
            trajectory: Actual sequence of agent actions
            expected_trajectory: Expected sequence of actions
            metadata: Additional context and information
            
        Returns:
            Unique interaction ID for tracking
        """
        interaction_id = self._generate_interaction_id(user_input, agent_output)
        
        # Estimate token usage (simplified - would use actual tokenizer in production)
        input_tokens = len(user_input.split())
        output_tokens = len(agent_output.split())
        
        # Estimate cost based on token usage (example rates)
        cost_per_1k_input = 0.0005  # Example rate
        cost_per_1k_output = 0.0015  # Example rate
        cost_usd = (input_tokens * cost_per_1k_input / 1000) + (output_tokens * cost_per_1k_output / 1000)
        
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            timestamp=datetime.now(),
            user_input=user_input,
            agent_output=agent_output,
            expected_output=expected_output,
            trajectory=trajectory or [],
            expected_trajectory=expected_trajectory or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            metadata=metadata or {}
        )
        
        self.interactions.append(interaction)
        self.logger.info(f"Recorded interaction {interaction_id}")
        
        return interaction_id
    
    def _generate_interaction_id(self, user_input: str, agent_output: str) -> str:
        """Generate unique interaction ID."""
        content = f"{user_input}{agent_output}{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    async def evaluate_response_accuracy(self, interaction_id: str) -> EvaluationResult:
        """
        Evaluate response accuracy using multiple sophisticated methods.
        
        Args:
            interaction_id: ID of the interaction to evaluate
            
        Returns:
            EvaluationResult with accuracy assessment
        """
        interaction = self._get_interaction(interaction_id)
        if not interaction or not interaction.expected_output:
            return EvaluationResult(
                interaction_id=interaction_id,
                metric_type=EvaluationMetric.ACCURACY,
                score=0.0,
                details={"error": "No expected output available for comparison"}
            )
        
        # Multiple accuracy assessment methods
        scores = {}
        
        # 1. Exact match (basic)
        exact_match = 1.0 if interaction.agent_output.strip().lower() == interaction.expected_output.strip().lower() else 0.0
        scores["exact_match"] = exact_match
        
        # 2. Token overlap (improved)
        agent_tokens = set(interaction.agent_output.lower().split())
        expected_tokens = set(interaction.expected_output.lower().split())
        
        if len(expected_tokens) > 0:
            precision = len(agent_tokens & expected_tokens) / len(agent_tokens) if agent_tokens else 0.0
            recall = len(agent_tokens & expected_tokens) / len(expected_tokens)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            scores.update({"precision": precision, "recall": recall, "f1_score": f1_score})
        
        # 3. Length similarity
        length_ratio = min(len(interaction.agent_output), len(interaction.expected_output)) / max(len(interaction.agent_output), len(interaction.expected_output), 1)
        scores["length_similarity"] = length_ratio
        
        # 4. Semantic similarity (simplified - would use embeddings in production)
        semantic_score = self._calculate_semantic_similarity(interaction.agent_output, interaction.expected_output)
        scores["semantic_similarity"] = semantic_score
        
        # Composite accuracy score
        composite_score = (scores.get("f1_score", 0) * 0.4 + 
                         scores.get("semantic_similarity", 0) * 0.4 +
                         scores.get("length_similarity", 0) * 0.2)
        
        return EvaluationResult(
            interaction_id=interaction_id,
            metric_type=EvaluationMetric.ACCURACY,
            score=composite_score,
            details=scores
        )
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (simplified implementation)."""
        # Simplified Jaccard similarity on word level
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def evaluate_trajectory_match(self, interaction_id: str, match_type: TrajectoryMatchType = TrajectoryMatchType.IN_ORDER_MATCH) -> EvaluationResult:
        """
        Evaluate agent trajectory against expected sequence of actions.
        
        Args:
            interaction_id: ID of the interaction to evaluate
            match_type: Type of trajectory matching to perform
            
        Returns:
            EvaluationResult with trajectory assessment
        """
        interaction = self._get_interaction(interaction_id)
        if not interaction or not interaction.expected_trajectory:
            return EvaluationResult(
                interaction_id=interaction_id,
                metric_type=EvaluationMetric.TRAJECTORY_MATCH,
                score=0.0,
                details={"error": "No expected trajectory available"}
            )
        
        actual = interaction.trajectory
        expected = interaction.expected_trajectory
        
        score = 0.0
        details = {"match_type": match_type.value, "actual": actual, "expected": expected}
        
        if match_type == TrajectoryMatchType.EXACT_MATCH:
            score = 1.0 if actual == expected else 0.0
            details["exact_match"] = score == 1.0
            
        elif match_type == TrajectoryMatchType.IN_ORDER_MATCH:
            # Check if expected actions appear in order (allowing extra actions)
            expected_idx = 0
            for action in actual:
                if expected_idx < len(expected) and action == expected[expected_idx]:
                    expected_idx += 1
            score = expected_idx / len(expected) if expected else 1.0
            details["matched_actions"] = expected_idx
            
        elif match_type == TrajectoryMatchType.ANY_ORDER_MATCH:
            # Check if all expected actions are present (any order)
            expected_set = set(expected)
            actual_set = set(actual)
            matched = len(expected_set & actual_set)
            score = matched / len(expected_set) if expected_set else 1.0
            details["matched_actions"] = matched
            
        elif match_type == TrajectoryMatchType.PRECISION_RECALL:
            expected_set = set(expected)
            actual_set = set(actual)
            
            if actual_set:
                precision = len(expected_set & actual_set) / len(actual_set)
            else:
                precision = 0.0
                
            if expected_set:
                recall = len(expected_set & actual_set) / len(expected_set)
            else:
                recall = 1.0
                
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            score = f1
            details.update({"precision": precision, "recall": recall, "f1_score": f1})
            
        elif match_type == TrajectoryMatchType.SINGLE_TOOL:
            # Check for presence of a specific critical action
            critical_action = expected[0] if expected else None
            score = 1.0 if critical_action and critical_action in actual else 0.0
            details["critical_action"] = critical_action
            details["found"] = score == 1.0
        
        return EvaluationResult(
            interaction_id=interaction_id,
            metric_type=EvaluationMetric.TRAJECTORY_MATCH,
            score=score,
            details=details
        )
    
    async def evaluate_with_llm_judge(self, interaction_id: str, evaluation_criteria: str) -> EvaluationResult:
        """
        Evaluate agent response using LLM-as-a-Judge methodology.
        
        Args:
            interaction_id: ID of the interaction to evaluate
            evaluation_criteria: Specific criteria for LLM evaluation
            
        Returns:
            EvaluationResult with LLM-based assessment
        """
        if not self.judge_enabled:
            return EvaluationResult(
                interaction_id=interaction_id,
                metric_type=EvaluationMetric.HELPFULNESS,
                score=0.0,
                details={"error": "LLM Judge not available"}
            )
        
        interaction = self._get_interaction(interaction_id)
        if not interaction:
            return EvaluationResult(
                interaction_id=interaction_id,
                metric_type=EvaluationMetric.HELPFULNESS,
                score=0.0,
                details={"error": "Interaction not found"}
            )
        
        # Construct evaluation prompt
        evaluation_prompt = f"""
You are an expert AI evaluator. Please evaluate the following agent response based on these criteria:

{evaluation_criteria}

**User Input:** {interaction.user_input}
**Agent Response:** {interaction.agent_output}

Please provide your evaluation in the following JSON format:
{{
  "overall_score": <1-5 integer score>,
  "rationale": "<detailed explanation>",
  "detailed_feedback": ["<specific feedback point 1>", "<point 2>"],
  "concerns": ["<concern 1>", "<concern 2>"],
  "recommended_action": "<recommendation>"
}}
"""
        
        try:
            response = self.judge_model.generate_content(
                evaluation_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            if not response.parts:
                raise ValueError("Empty response from LLM Judge")
                
            evaluation_data = json.loads(response.text)
            evaluation = LLMJudgeEvaluation.model_validate(evaluation_data)
            
            # Normalize score to 0-1 range
            normalized_score = (evaluation.overall_score - 1) / 4.0
            
            return EvaluationResult(
                interaction_id=interaction_id,
                metric_type=EvaluationMetric.HELPFULNESS,
                score=normalized_score,
                details={
                    "raw_score": evaluation.overall_score,
                    "rationale": evaluation.rationale,
                    "detailed_feedback": evaluation.detailed_feedback,
                    "concerns": evaluation.concerns,
                    "recommended_action": evaluation.recommended_action
                }
            )
            
        except Exception as e:
            self.logger.error(f"LLM Judge evaluation failed: {e}")
            return EvaluationResult(
                interaction_id=interaction_id,
                metric_type=EvaluationMetric.HELPFULNESS,
                score=0.0,
                details={"error": f"LLM evaluation failed: {str(e)}"}
            )
    
    async def monitor_performance_trends(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Monitor agent performance trends over a specified time window.
        
        Args:
            window_hours: Time window in hours for trend analysis
            
        Returns:
            Dictionary containing trend analysis and alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_interactions = [i for i in self.interactions if i.timestamp >= cutoff_time]
        recent_evaluations = [e for e in self.evaluation_results if e.timestamp >= cutoff_time]
        
        if not recent_interactions:
            return {"error": "No recent interactions to analyze"}
        
        # Analyze trends by metric type
        trends = {}
        alerts = []
        
        for metric_type in EvaluationMetric:
            metric_results = [e for e in recent_evaluations if e.metric_type == metric_type]
            if metric_results:
                scores = [e.score for e in metric_results]
                trends[metric_type.value] = {
                    "count": len(scores),
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores)
                }
                
                # Check for performance degradation
                if metric_type.value in self.performance_baselines:
                    baseline = self.performance_baselines[metric_type.value]
                    current_mean = trends[metric_type.value]["mean"]
                    degradation = (baseline - current_mean) / baseline if baseline > 0 else 0
                    
                    if degradation > 0.1:  # 10% degradation threshold
                        alerts.append({
                            "type": "performance_degradation",
                            "metric": metric_type.value,
                            "baseline": baseline,
                            "current": current_mean,
                            "degradation_pct": degradation * 100
                        })
        
        # Resource utilization trends
        total_tokens = sum(i.input_tokens + i.output_tokens for i in recent_interactions)
        total_cost = sum(i.cost_usd for i in recent_interactions)
        avg_latency = statistics.mean([i.latency_ms for i in recent_interactions if i.latency_ms > 0]) if any(i.latency_ms > 0 for i in recent_interactions) else 0
        
        resource_trends = {
            "total_interactions": len(recent_interactions),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_latency_ms": avg_latency,
            "tokens_per_interaction": total_tokens / len(recent_interactions),
            "cost_per_interaction": total_cost / len(recent_interactions)
        }
        
        # Check for resource usage anomalies
        if total_cost > self.anomaly_thresholds.get("cost_threshold", float('inf')):
            alerts.append({
                "type": "high_cost_usage",
                "current_cost": total_cost,
                "threshold": self.anomaly_thresholds["cost_threshold"]
            })
        
        return {
            "window_hours": window_hours,
            "performance_trends": trends,
            "resource_trends": resource_trends,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_anomalies(self, interaction_id: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a specific agent interaction.
        
        Args:
            interaction_id: ID of the interaction to analyze
            
        Returns:
            List of detected anomalies with details
        """
        interaction = self._get_interaction(interaction_id)
        if not interaction:
            return [{"error": "Interaction not found"}]
        
        anomalies = []
        
        # Check for unusual response length
        avg_output_length = statistics.mean([len(i.agent_output) for i in self.interactions[-50:]])  # Last 50 interactions
        if len(interaction.agent_output) > avg_output_length * 3:
            anomalies.append({
                "type": "unusually_long_response",
                "current_length": len(interaction.agent_output),
                "average_length": avg_output_length
            })
        
        # Check for high latency
        if interaction.latency_ms > 10000:  # 10 second threshold
            anomalies.append({
                "type": "high_latency",
                "latency_ms": interaction.latency_ms,
                "threshold": 10000
            })
        
        # Check for high token usage
        total_tokens = interaction.input_tokens + interaction.output_tokens
        avg_tokens = statistics.mean([(i.input_tokens + i.output_tokens) for i in self.interactions[-50:]])
        if total_tokens > avg_tokens * 2:
            anomalies.append({
                "type": "high_token_usage",
                "current_tokens": total_tokens,
                "average_tokens": avg_tokens
            })
        
        # Check for empty or minimal responses
        if len(interaction.agent_output.strip()) < 10:
            anomalies.append({
                "type": "minimal_response",
                "response_length": len(interaction.agent_output.strip())
            })
        
        return anomalies
    
    def _get_interaction(self, interaction_id: str) -> Optional[AgentInteraction]:
        """Retrieve interaction by ID."""
        for interaction in self.interactions:
            if interaction.interaction_id == interaction_id:
                return interaction
        return None
    
    async def run_comprehensive_evaluation(self, interaction_id: str) -> Dict[str, EvaluationResult]:
        """
        Run comprehensive evaluation across all available metrics.
        
        Args:
            interaction_id: ID of the interaction to evaluate comprehensively
            
        Returns:
            Dictionary of evaluation results by metric type
        """
        results = {}
        
        # Run all available evaluations
        evaluations = [
            ("accuracy", self.evaluate_response_accuracy(interaction_id)),
            ("trajectory", self.evaluate_trajectory_match(interaction_id)),
            ("helpfulness", self.evaluate_with_llm_judge(
                interaction_id, 
                "Evaluate the response for helpfulness, clarity, accuracy, and completeness."
            ))
        ]
        
        for name, eval_coroutine in evaluations:
            try:
                result = await eval_coroutine
                results[name] = result
                self.evaluation_results.append(result)
            except Exception as e:
                self.logger.error(f"Evaluation '{name}' failed: {e}")
                results[name] = EvaluationResult(
                    interaction_id=interaction_id,
                    metric_type=EvaluationMetric.ACCURACY,  # Default
                    score=0.0,
                    details={"error": str(e)}
                )
        
        return results
    
    def generate_evaluation_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for specified time window.
        
        Args:
            time_window_hours: Hours to include in the report
            
        Returns:
            Comprehensive evaluation report
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_interactions = [i for i in self.interactions if i.timestamp >= cutoff_time]
        recent_evaluations = [e for e in self.evaluation_results if e.timestamp >= cutoff_time]
        
        if not recent_interactions:
            return {"error": "No data available for report generation"}
        
        # Overall statistics
        total_interactions = len(recent_interactions)
        total_evaluations = len(recent_evaluations)
        
        # Performance metrics summary
        performance_summary = {}
        for metric_type in EvaluationMetric:
            metric_results = [e for e in recent_evaluations if e.metric_type == metric_type]
            if metric_results:
                scores = [e.score for e in metric_results]
                performance_summary[metric_type.value] = {
                    "average_score": statistics.mean(scores),
                    "best_score": max(scores),
                    "worst_score": min(scores),
                    "total_evaluations": len(scores)
                }
        
        # Resource utilization
        resource_summary = {
            "total_tokens_used": sum(i.input_tokens + i.output_tokens for i in recent_interactions),
            "total_cost_usd": sum(i.cost_usd for i in recent_interactions),
            "average_latency_ms": statistics.mean([i.latency_ms for i in recent_interactions if i.latency_ms > 0]) if any(i.latency_ms > 0 for i in recent_interactions) else 0
        }
        
        # Anomaly detection summary
        total_anomalies = 0
        anomaly_types = defaultdict(int)
        for interaction in recent_interactions:
            interaction_anomalies = self.detect_anomalies(interaction.interaction_id)
            total_anomalies += len(interaction_anomalies)
            for anomaly in interaction_anomalies:
                if "type" in anomaly:
                    anomaly_types[anomaly["type"]] += 1
        
        return {
            "report_period": {
                "start_time": cutoff_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_hours": time_window_hours
            },
            "summary": {
                "total_interactions": total_interactions,
                "total_evaluations": total_evaluations,
                "total_anomalies": total_anomalies
            },
            "performance_metrics": performance_summary,
            "resource_utilization": resource_summary,
            "anomaly_breakdown": dict(anomaly_types),
            "recommendations": self._generate_recommendations(performance_summary, resource_summary, anomaly_types)
        }
    
    def _generate_recommendations(self, performance: Dict, resources: Dict, anomalies: Dict) -> List[str]:
        """Generate actionable recommendations based on evaluation data."""
        recommendations = []
        
        # Performance recommendations
        for metric, data in performance.items():
            if data["average_score"] < 0.7:
                recommendations.append(f"Consider improving {metric} - current average: {data['average_score']:.2f}")
        
        # Resource recommendations
        if resources.get("average_latency_ms", 0) > 5000:
            recommendations.append("High latency detected - consider optimizing response generation")
        
        # Anomaly recommendations
        if "high_token_usage" in anomalies:
            recommendations.append("Frequent high token usage - review prompt efficiency")
        
        if not recommendations:
            recommendations.append("System performance appears healthy - continue monitoring")
        
        return recommendations

# --- Token Usage Monitoring ---
class LLMInteractionMonitor:
    """Monitor and track LLM token usage for cost management."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.interaction_log = []
        self.cost_tracker = {
            "total_cost": 0.0,
            "daily_cost": defaultdict(float),
            "model_costs": defaultdict(float)
        }
    
    def record_interaction(self, prompt: str, response: str, model_name: str = "default"):
        """Record an LLM interaction with token and cost tracking."""
        # Simplified token counting - use actual tokenizer in production
        input_tokens = len(prompt.split())
        output_tokens = len(response.split())
        
        # Estimate costs (example rates)
        cost_per_1k_input = 0.0005
        cost_per_1k_output = 0.0015
        interaction_cost = (input_tokens * cost_per_1k_input / 1000) + (output_tokens * cost_per_1k_output / 1000)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Update cost tracking
        self.cost_tracker["total_cost"] += interaction_cost
        today = datetime.now().date().isoformat()
        self.cost_tracker["daily_cost"][today] += interaction_cost
        self.cost_tracker["model_costs"][model_name] += interaction_cost
        
        # Log interaction
        self.interaction_log.append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": interaction_cost
        })
        
        print(f"Recorded interaction: Input=\\{input_tokens}, Output=\\{output_tokens}, Cost=\\$\\{interaction_cost:.6f}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage and cost summary."""
        return {
            "token_usage": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens
            },
            "cost_summary": self.cost_tracker,
            "recent_interactions": len(self.interaction_log),
            "avg_tokens_per_interaction": (self.total_input_tokens + self.total_output_tokens) / max(len(self.interaction_log), 1)
        }

# --- Demonstration Functions ---
async def demonstrate_comprehensive_evaluation():
    """Demonstrate comprehensive agent evaluation and monitoring."""
    
    print("📊 Comprehensive Agent Evaluation and Monitoring Demonstration")
    print("=" * 75)
    
    # Initialize evaluation system
    evaluator = ComprehensiveAgentEvaluator()
    token_monitor = LLMInteractionMonitor()
    
    # Simulate agent interactions
    test_cases = [
        {
            "user_input": "What is the capital of France?",
            "agent_output": "The capital of France is Paris.",
            "expected_output": "Paris is the capital of France.",
            "trajectory": ["knowledge_lookup", "factual_response"],
            "expected_trajectory": ["knowledge_lookup", "factual_response"]
        },
        {
            "user_input": "Help me plan a vacation to Japan",
            "agent_output": "I'd be happy to help you plan a vacation to Japan! Japan offers incredible experiences from bustling Tokyo to serene Kyoto temples. What type of activities interest you most?",
            "expected_output": "Japan is a wonderful destination with diverse attractions.",
            "trajectory": ["intent_analysis", "knowledge_retrieval", "personalized_response"],
            "expected_trajectory": ["intent_analysis", "recommendation_generation", "personalized_response"]
        },
        {
            "user_input": "Calculate the square root of 144",
            "agent_output": "The square root of 144 is 12.",
            "expected_output": "12",
            "trajectory": ["math_calculation", "direct_answer"],
            "expected_trajectory": ["math_calculation", "direct_answer"]
        }
    ]
    
    print("\\n🔄 Recording Agent Interactions")
    print("-" * 40)
    
    interaction_ids = []
    for i, case in enumerate(test_cases, 1):
        print(f"\\nCase \\{i}: \\{case['user_input'][:50]}...")
        
        # Record interaction
        interaction_id = await evaluator.record_interaction(
            user_input=case["user_input"],
            agent_output=case["agent_output"],
            expected_output=case["expected_output"],
            trajectory=case["trajectory"],
            expected_trajectory=case["expected_trajectory"],
            metadata={"test_case": i}
        )
        
        interaction_ids.append(interaction_id)
        
        # Record token usage
        token_monitor.record_interaction(
            prompt=case["user_input"],
            response=case["agent_output"],
            model_name="test-model-v1"
        )
        
        print(f"✅ Recorded interaction: \\{interaction_id}")
    
    print("\\n📋 Running Comprehensive Evaluations")
    print("-" * 40)
    
    for i, interaction_id in enumerate(interaction_ids, 1):
        print(f"\\nEvaluating Interaction {i}:")
        
        # Run comprehensive evaluation
        results = await evaluator.run_comprehensive_evaluation(interaction_id)
        
        for metric_name, result in results.items():
            print(f"  \\{metric_name.title()}: \\{result.score:.3f}")
            if "error" not in result.details:
                if metric_name == "accuracy" and "f1_score" in result.details:
                    print(f"    F1 Score: \\{result.details['f1_score']:.3f}")
                elif metric_name == "trajectory" and "matched_actions" in result.details:
                    print(f"    Matched Actions: \\{result.details['matched_actions']}")
                elif metric_name == "helpfulness" and "rationale" in result.details:
                    print(f"    Rationale: \\{result.details['rationale'][:100]}...")
        
        # Check for anomalies
        anomalies = evaluator.detect_anomalies(interaction_id)
        if anomalies and "error" not in anomalies[0]:
            print(f"  ⚠️  Anomalies detected: \\{len(anomalies)}")
            for anomaly in anomalies:
                print(f"    - \\{anomaly.get('type', 'Unknown')}")
    
    print("\\n📈 Performance Monitoring and Trends")
    print("-" * 40)
    
    # Monitor performance trends
    trends = await evaluator.monitor_performance_trends(window_hours=1)
    
    if "performance_trends" in trends:
        print("Performance Trends:")
        for metric, data in trends["performance_trends"].items():
            print(f"  \\{metric}: Mean=\\{data['mean']:.3f}, Count=\\{data['count']}")
    
    if trends.get("alerts"):
        print("\\n⚠️  Alerts:")
        for alert in trends["alerts"]:
            print(f"  - \\{alert['type']}: \\{alert}")
    
    # Resource utilization
    if "resource_trends" in trends:
        resources = trends["resource_trends"]
        print(f"\\nResource Usage:")
        print(f"  Total interactions: \\{resources['total_interactions']}")
        print(f"  Total tokens: \\{resources['total_tokens']}")
        print(f"  Total cost: \\$\\{resources['total_cost_usd']:.6f}")
        print(f"  Average latency: \\{resources['avg_latency_ms']:.1f}ms")
    
    print("\\n💰 Token Usage and Cost Analysis")
    print("-" * 40)
    
    usage_summary = token_monitor.get_usage_summary()
    token_usage = usage_summary["token_usage"]
    cost_summary = usage_summary["cost_summary"]
    
    print(f"Token Usage:")
    print(f"  Input tokens: \\{token_usage['total_input']:,}")
    print(f"  Output tokens: \\{token_usage['total_output']:,}")
    print(f"  Total tokens: \\{token_usage['total_tokens']:,}")
    
    print(f"\\nCost Analysis:")
    print(f"  Total cost: \\$\\{cost_summary['total_cost']:.6f}")
    print(f"  Average per interaction: \\$\\{cost_summary['total_cost'] / len(test_cases):.6f}")
    
    print("\\n📊 Comprehensive Evaluation Report")
    print("-" * 40)
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(time_window_hours=1)
    
    if "error" not in report:
        summary = report["summary"]
        print(f"Report Summary:")
        print(f"  Interactions: \\{summary['total_interactions']}")
        print(f"  Evaluations: \\{summary['total_evaluations']}")
        print(f"  Anomalies: \\{summary['total_anomalies']}")
        
        if report.get("recommendations"):
            print(f"\\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  - \\{rec}")
    
    print("\\n✅ Comprehensive Evaluation Demonstration Complete!")
    print("Advanced monitoring and evaluation systems enable continuous quality assurance")

# Example usage
if __name__ == "__main__":
    asyncio.run(demonstrate_comprehensive_evaluation())`,
  sections: [
    {
      title: 'Performance Tracking and Monitoring',
      content: `Comprehensive agent evaluation begins with systematic performance tracking across multiple dimensions of operational effectiveness.

**Response Accuracy Assessment**

Beyond simple exact matching, sophisticated accuracy evaluation employs multiple methodologies to assess response quality:

• **Token-based Analysis**: Precision, recall, and F1 scores based on token overlap between actual and expected responses
• **Semantic Similarity**: Embedding-based comparison to capture meaning rather than just literal matches
• **Length Similarity**: Appropriate response length relative to expected outputs
• **Domain-specific Metrics**: Tailored evaluation criteria for specialized applications

**Latency and Performance Monitoring**

Real-time performance monitoring tracks critical operational metrics:

• **Response Latency**: End-to-end processing time from input to final response
• **Resource Consumption**: CPU, memory, and GPU utilization during agent operations
• **Throughput Metrics**: Requests handled per unit time under varying load conditions
• **Availability Tracking**: System uptime and error rates across different operational scenarios

**Token Usage and Cost Optimization**

For LLM-powered agents, detailed token tracking enables cost management and resource optimization:

• **Input/Output Token Monitoring**: Separate tracking of prompt tokens versus generated response tokens
• **Cost Attribution**: Per-interaction cost calculation based on model-specific pricing
• **Usage Patterns**: Analysis of token consumption trends and optimization opportunities
• **Budget Management**: Real-time cost tracking against allocated budgets with alerting thresholds

**Example Implementation:**

\`\`\`python
def evaluate_response_accuracy(agent_output: str, expected_output: str) -> float:
    # Multiple evaluation approaches
    exact_match = 1.0 if agent_output.strip().lower() == expected_output.strip().lower() else 0.0
    
    # Token-based F1 score
    agent_tokens = set(agent_output.lower().split())
    expected_tokens = set(expected_output.lower().split())
    
    precision = len(agent_tokens & expected_tokens) / len(agent_tokens) if agent_tokens else 0.0
    recall = len(agent_tokens & expected_tokens) / len(expected_tokens) if expected_tokens else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1_score
\`\`\`

This comprehensive approach provides nuanced assessment beyond binary correct/incorrect evaluations.`
    },
    {
      title: 'Agent Trajectory Analysis',
      content: `Agent trajectory evaluation examines the sequence of actions and decisions leading to final outputs, providing insight into the reasoning and problem-solving process.

**Trajectory Matching Methodologies**

Different matching strategies assess trajectory quality based on specific requirements:

**Exact Match**: Perfect sequence alignment requiring identical action ordering
- Use case: Critical safety systems where precise procedures are mandatory
- Example: Medical diagnostic protocols with required validation steps

**In-Order Match**: Expected actions must appear in correct sequence, allowing additional steps
- Use case: Flexible workflows where extra validation or exploration is acceptable
- Example: Customer service agents that may gather additional context

**Any-Order Match**: All expected actions present regardless of sequence
- Use case: Independent tasks where order doesn't affect outcome
- Example: Data collection agents gathering information from multiple sources

**Precision/Recall Analysis**: Comprehensive assessment of action relevance and completeness
- Precision: Proportion of agent actions that were necessary/expected
- Recall: Proportion of expected actions that were actually performed
- F1 Score: Balanced measure combining precision and recall

**Single-Tool Verification**: Confirmation of critical action execution
- Use case: Ensuring essential steps are completed regardless of other actions
- Example: Security agents must perform authorization checks

**Implementation Example:**

\`\`\`python
def evaluate_trajectory_match(actual: List[str], expected: List[str], match_type: str) -> float:
    if match_type == "exact_match":
        return 1.0 if actual == expected else 0.0
    
    elif match_type == "in_order_match":
        expected_idx = 0
        for action in actual:
            if expected_idx < len(expected) and action == expected[expected_idx]:
                expected_idx += 1
        return expected_idx / len(expected) if expected else 1.0
    
    elif match_type == "precision_recall":
        expected_set = set(expected)
        actual_set = set(actual)
        
        precision = len(expected_set & actual_set) / len(actual_set) if actual_set else 0.0
        recall = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
\`\`\`

**Error Pattern Detection**

Trajectory analysis enables identification of common failure modes:
• **Infinite Loops**: Agents stuck in repetitive action cycles
• **Premature Termination**: Missing critical final validation steps
• **Tool Misuse**: Inappropriate tool selection for specific tasks
• **Context Loss**: Forgetting important information between actions`
    },
    {
      title: 'LLM-as-a-Judge Evaluation',
      content: `LLM-as-a-Judge methodology leverages advanced language models to evaluate subjective qualities that traditional metrics cannot capture effectively.

**Evaluation Framework Design**

Comprehensive evaluation requires structured rubrics that define clear assessment criteria:

**Multi-Dimensional Scoring**: 
- Clarity and Precision (1-5): Language clarity and technical accuracy
- Neutrality and Bias (1-5): Objective presentation without leading language
- Relevance and Focus (1-5): Direct alignment with query requirements
- Completeness (1-5): Comprehensive coverage of necessary information
- Audience Appropriateness (1-5): Proper complexity level and terminology

**Structured Output Requirements**:
- Overall Score: Composite assessment from 1-5
- Detailed Rationale: Explanation of scoring decisions
- Specific Feedback: Actionable improvement suggestions
- Concerns: Identification of potential issues or risks
- Recommended Actions: Next steps for improvement

**Domain-Specific Adaptations**

Different domains require specialized evaluation criteria:

**Legal Domain Evaluation**:
\`\`\`python
LEGAL_SURVEY_RUBRIC = \"\"\"
You are an expert legal survey methodologist evaluating question quality.

Criteria:
1. Legal Terminology Accuracy (1-5): Proper use of legal concepts
2. Jurisdictional Appropriateness (1-5): Relevant to applicable legal framework  
3. Ethical Compliance (1-5): Adherence to professional standards
4. Clarity for Target Audience (1-5): Appropriate for respondent expertise
5. Bias Prevention (1-5): Neutral, non-leading question structure
\"\"\"
\`\`\`

**Technical Documentation Evaluation**:
- Accuracy: Technical correctness and up-to-date information
- Completeness: Coverage of all necessary implementation details
- Clarity: Accessibility to intended audience skill level
- Examples: Quality and relevance of provided code samples

**Customer Service Evaluation**:
- Helpfulness: Practical value of response to user query
- Empathy: Appropriate emotional tone and understanding
- Resolution: Effectiveness in addressing user concerns
- Professionalism: Adherence to company communication standards

**Implementation Considerations**

Effective LLM-as-a-Judge implementation requires careful attention to:

**Temperature Settings**: Lower temperature (0.1-0.3) for consistent, deterministic evaluations
**Model Selection**: Balance between capability and cost (e.g., GPT-4 for quality vs GPT-3.5 for speed)
**Prompt Engineering**: Detailed rubrics with examples improve evaluation consistency
**Output Validation**: JSON schema enforcement ensures structured, parseable results
**Bias Mitigation**: Regular evaluation of judge consistency across different content types`
    },
    {
      title: 'Advanced Evaluation Frameworks',
      content: `Modern agent evaluation extends beyond individual performance to encompass multi-agent systems, contractor frameworks, and comprehensive monitoring ecosystems.

**Multi-Agent System Evaluation**

Evaluating distributed agent systems requires assessment of both individual and collective performance:

**Collaboration Effectiveness**:
- Information Handoff Quality: Accuracy and completeness of inter-agent communication
- Task Coordination: Proper sequencing and dependency management
- Resource Sharing: Efficient utilization of shared computational resources
- Conflict Resolution: Handling of competing priorities and resource constraints

**System-Level Metrics**:
- Overall Goal Achievement: Success rate in completing complex, multi-step objectives
- Scalability Performance: System behavior under increasing agent counts
- Fault Tolerance: Graceful degradation when individual agents fail
- Communication Overhead: Network and processing costs of inter-agent coordination

**Contractor-Based Agent Evolution**

The evolution from simple agents to formal "contractors" introduces new evaluation paradigms:

**Contract Specification Quality**:
- Completeness: All requirements, deliverables, and constraints clearly defined
- Verifiability: Objective criteria for success measurement
- Feasibility: Realistic scope given available resources and capabilities

**Negotiation Process Assessment**:
- Requirement Clarification: Agent ability to identify and resolve ambiguities
- Alternative Proposal Generation: Creative problem-solving when constraints are encountered
- Stakeholder Communication: Clear explanation of limitations and alternatives

**Iterative Quality Improvement**:
- Self-Validation Loops: Internal quality checks before deliverable submission
- Continuous Refinement: Progressive improvement through multiple iteration cycles
- Stakeholder Feedback Integration: Responsive incorporation of user guidance

**Hierarchical Task Decomposition**:
- Subcontract Generation: Quality of task breakdown into manageable components
- Dependency Management: Proper identification and handling of task interdependencies
- Project Coordination: Effective management of complex, multi-agent projects

**Google ADK Integration**

Google's Agent Development Kit provides structured evaluation methodologies:

**Test File Structure**: Simple agent-model interactions for unit testing
- Single session focus with multiple user-agent turns
- Expected tool usage trajectory specification
- Reference response definition for comparison

**Evalset Framework**: Complex, multi-turn conversation simulation
- Multiple session management for integration testing
- Comprehensive interaction pattern coverage
- Scalable evaluation across different agent configurations

**Evaluation Execution Methods**:
- Web-based UI: Interactive session creation and real-time evaluation
- Pytest Integration: Automated testing within CI/CD pipelines
- Command-line Interface: Batch evaluation for regular monitoring

**Example ADK Integration**:
\`\`\`python
# Pytest-based evaluation
from adk.evaluation import AgentEvaluator

def test_agent_performance():
    evaluator = AgentEvaluator()
    results = evaluator.evaluate(
        agent_module="my_agent.py",
        test_file="test_cases.json"
    )
    assert results.success_rate > 0.85
\`\`\`

This integrated approach ensures comprehensive coverage from individual interactions to system-wide performance assessment.`
    }
  ],
  practicalApplications: [
    'Performance tracking in live systems monitoring accuracy, latency, and resource consumption of production agents with real-time alerting',
    'A/B testing frameworks comparing different agent versions, algorithms, or model configurations to identify optimal approaches systematically',
    'Compliance and safety audits generating automated reports tracking ethical guidelines, regulatory requirements, and safety protocol adherence',
    'Drift detection systems monitoring agent output relevance and accuracy over time to identify performance degradation from environmental changes',
    'Anomaly detection in agent behavior identifying unusual actions that might indicate errors, attacks, or emergent undesired behaviors',
    'Learning progress assessment for adaptive agents tracking improvement curves, skill development, and generalization capabilities across tasks',
    'Multi-agent system coordination evaluation measuring collaborative effectiveness, communication quality, and distributed task success rates',
    'Cost optimization through detailed token usage monitoring and resource allocation analysis for LLM-powered agent deployments'
  ],
  practicalExamples: [
    {
      title: 'Enterprise Customer Service Agent Evaluation',
      description: 'Comprehensive evaluation system for customer service agents measuring response accuracy, helpfulness, resolution effectiveness, and compliance with company policies.',
      implementation: 'Multi-dimensional LLM-as-a-Judge evaluation with domain-specific rubrics, trajectory analysis for interaction flow assessment, real-time performance monitoring, and automated A/B testing for continuous improvement.'
    },
    {
      title: 'Financial Advisory Agent Monitoring',
      description: 'Specialized evaluation framework for financial advisory agents ensuring regulatory compliance, accuracy of recommendations, and appropriate risk assessment.',
      implementation: 'Compliance-focused evaluation with regulatory requirement tracking, accuracy assessment against market data, risk appropriateness scoring, and detailed audit trail generation for regulatory reporting.'
    },
    {
      title: 'Multi-Agent Research System Assessment',
      description: 'Evaluation framework for distributed research agents collaborating on complex information gathering and analysis tasks across multiple domains.',
      implementation: 'Collaborative effectiveness measurement, information quality assessment, task coordination evaluation, resource utilization optimization, and comprehensive trajectory analysis across multiple specialized agents.'
    }
  ],
  nextSteps: [
    'Implement basic performance tracking for accuracy, latency, and resource consumption in your agent applications',
    'Deploy comprehensive evaluation frameworks using LLM-as-a-Judge methodology for subjective quality assessment',
    'Establish agent trajectory analysis to understand and optimize decision-making processes and tool usage patterns',
    'Create automated monitoring systems with alerting for performance degradation, anomalies, and resource threshold violations',
    'Develop A/B testing infrastructure to systematically compare agent versions and optimization strategies',
    'Integrate evaluation frameworks with CI/CD pipelines for continuous quality assurance during development cycles',
    'Implement multi-agent evaluation systems focusing on collaboration effectiveness and distributed task success',
    'Build contractor-based agent frameworks with formal agreements, negotiation capabilities, and hierarchical task management'
  ],
  references: [
    'ADK Web: https://github.com/google/adk-web',
    'ADK Evaluate: https://google.github.io/adk-docs/evaluate/',
    'Survey on Evaluation of LLM-based Agents: https://arxiv.org/abs/2503.16416',
    'Agent-as-a-Judge: Evaluate Agents with Agents: https://arxiv.org/abs/2410.10934',
    'Agent Companion (gulli et al): https://www.kaggle.com/whitepaper-agent-companion',
    'Google AI Evaluation Best Practices: https://cloud.google.com/vertex-ai/docs/evaluation',
    'OpenAI Evaluation Framework: https://platform.openai.com/docs/guides/evals'
  ],
  navigation: {
    previous: { href: '/chapters/guardrails-safety-patterns', title: 'Guardrails / Safety Patterns' },
    next: { href: '/chapters/prioritization', title: 'Prioritization' }
  }
};
