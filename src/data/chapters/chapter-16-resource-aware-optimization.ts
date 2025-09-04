import { Chapter } from '../types'

export const resourceAwareOptimizationChapter: Chapter = {
  id: 'resource-aware-optimization',
  number: 16,
  title: 'Resource-Aware Optimization',
  part: 'Part Four – Scaling, Safety, and Discovery',
  description: 'Enable intelligent agents to dynamically monitor and manage computational, temporal, and financial resources through smart model selection, adaptive tool use, and cost-effective optimization strategies.',
  readingTime: '33 min read',
  difficulty: 'Advanced',
  content: {
    overview: `Resource-Aware Optimization enables intelligent agents to dynamically monitor and manage computational, temporal, and financial resources during operation, making strategic decisions about resource allocation to achieve goals within specified budgets or optimize efficiency. This differs from simple planning, which primarily focuses on action sequencing, by requiring agents to make real-time decisions regarding execution strategies based on resource constraints and optimization objectives.

This pattern involves choosing between more accurate but expensive models and faster, lower-cost alternatives, deciding whether to allocate additional compute for refined responses versus returning quicker answers, and implementing fallback mechanisms that ensure graceful degradation when preferred resources are unavailable. For example, an agent analyzing financial data might use a faster, affordable model for preliminary reports when immediate results are needed, but switch to a more powerful, precise model for critical investment decisions when accuracy is paramount and budget allows.

The core strategy involves implementing Router Agents that classify query complexity and route requests to appropriate models (such as Gemini Flash for simple queries and Gemini Pro for complex reasoning), combined with Critique Agents that evaluate response quality and provide feedback loops for continuous optimization. Advanced implementations include adaptive tool selection, contextual pruning to manage token costs, proactive resource prediction, and sophisticated fallback mechanisms that maintain service continuity during resource constraints.

Resource-Aware Optimization is essential for building scalable, cost-effective agent systems that can operate efficiently across varying workloads while maintaining appropriate quality levels and staying within operational budgets.`,

    keyPoints: [
      'Dynamic resource management enabling agents to balance computational, temporal, and financial constraints through intelligent model selection and adaptive execution strategies',
      'Router Agent architecture that classifies query complexity and routes requests to appropriate models based on resource availability and optimization objectives',
      'Critique Agent systems providing quality assurance, performance monitoring, and feedback loops for continuous optimization of routing decisions and resource allocation',
      'Fallback mechanisms ensuring graceful degradation and service continuity when preferred models are unavailable due to throttling, cost limits, or system overload',
      'Multi-model orchestration supporting seamless switching between cost-effective models (Gemini Flash) for simple tasks and powerful models (Gemini Pro) for complex reasoning',
      'Advanced optimization techniques including adaptive tool selection, contextual pruning, proactive resource prediction, and cost-sensitive exploration in multi-agent systems',
      'Integration with unified model interfaces (OpenRouter) enabling automated model selection, sequential fallback, and cost optimization across hundreds of AI models',
      'Comprehensive monitoring and analytics for tracking resource utilization, cost efficiency, performance metrics, and quality trade-offs across different optimization strategies'
    ],

    codeExample: `# Comprehensive Resource-Aware Optimization System
# Advanced implementation with Router Agents, Critique Systems, and Dynamic Model Selection

import os
import asyncio
import requests
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta
import statistics

# External dependencies (conceptual imports based on actual implementations)
try:
    from openai import OpenAI
    from google.adk.agents import Agent, BaseAgent
    from google.adk.events import Event
    from google.adk.agents.invocation_context import InvocationContext
    from dotenv import load_dotenv
except ImportError:
    print("Note: This example requires openai, google-adk, and python-dotenv packages")

# Load environment configuration
load_dotenv()

class QueryComplexity(Enum):
    """Query complexity classification levels."""
    SIMPLE = "simple"
    REASONING = "reasoning"
    INTERNET_SEARCH = "internet_search"
    COMPLEX_ANALYSIS = "complex_analysis"

class ModelTier(Enum):
    """Model performance and cost tiers."""
    FAST = "fast"          # Low cost, high speed (e.g., Gemini Flash)
    BALANCED = "balanced"   # Medium cost, balanced performance
    PREMIUM = "premium"     # High cost, maximum capability (e.g., Gemini Pro)

@dataclass
class ModelConfig:
    """Configuration for different model options."""
    name: str
    tier: ModelTier
    cost_per_token: float
    latency_ms: int
    capability_score: float
    max_tokens: int
    fallback_model: Optional[str] = None

@dataclass
class ResourceBudget:
    """Resource budget constraints and tracking."""
    max_cost_per_query: float = 1.0
    max_latency_ms: int = 30000
    daily_budget: float = 100.0
    current_daily_spend: float = 0.0
    query_count: int = 0
    
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if query can be executed within budget constraints."""
        return (self.current_daily_spend + estimated_cost) <= self.daily_budget
    
    def update_spend(self, actual_cost: float):
        """Update budget tracking with actual costs."""
        self.current_daily_spend += actual_cost
        self.query_count += 1

@dataclass
class OptimizationMetrics:
    """Performance and optimization tracking metrics."""
    total_queries: int = 0
    total_cost: float = 0.0
    average_latency: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    model_usage: Dict[str, int] = field(default_factory=dict)
    fallback_count: int = 0
    cost_savings: float = 0.0
    
    def add_query_result(self, cost: float, latency: float, quality: float, model_used: str):
        """Track results from completed query."""
        self.total_queries += 1
        self.total_cost += cost
        self.quality_scores.append(quality)
        
        # Update average latency using incremental calculation
        self.average_latency = ((self.average_latency * (self.total_queries - 1)) + latency) / self.total_queries
        
        # Track model usage
        self.model_usage[model_used] = self.model_usage.get(model_used, 0) + 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        return {
            "total_queries": self.total_queries,
            "total_cost": self.total_cost,
            "average_cost_per_query": self.total_cost / max(self.total_queries, 1),
            "average_latency_ms": self.average_latency,
            "average_quality": statistics.mean(self.quality_scores) if self.quality_scores else 0.0,
            "quality_std_dev": statistics.stdev(self.quality_scores) if len(self.quality_scores) > 1 else 0.0,
            "model_usage_distribution": self.model_usage,
            "fallback_rate": self.fallback_count / max(self.total_queries, 1),
            "estimated_cost_savings": self.cost_savings
        }

class ModelRegistry:
    """Registry of available models with their configurations and capabilities."""
    
    def __init__(self):
        self.models = {
            "gemini-flash": ModelConfig(
                name="gemini-2.0-flash-exp",
                tier=ModelTier.FAST,
                cost_per_token=0.00001,
                latency_ms=500,
                capability_score=0.7,
                max_tokens=8192,
                fallback_model="gpt-4o-mini"
            ),
            "gemini-pro": ModelConfig(
                name="gemini-2.0-pro-exp",
                tier=ModelTier.PREMIUM,
                cost_per_token=0.0001,
                latency_ms=2000,
                capability_score=0.95,
                max_tokens=32768,
                fallback_model="gemini-flash"
            ),
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                tier=ModelTier.PREMIUM,
                cost_per_token=0.00008,
                latency_ms=1800,
                capability_score=0.92,
                max_tokens=16384,
                fallback_model="gpt-4o-mini"
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                tier=ModelTier.FAST,
                cost_per_token=0.00002,
                latency_ms=800,
                capability_score=0.75,
                max_tokens=8192,
                fallback_model=None
            )
        }
        print(f"📊 Model Registry initialized with \\{len(self.models)} available models")
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Retrieve model configuration by ID."""
        return self.models.get(model_id)
    
    def get_models_by_tier(self, tier: ModelTier) -> List[ModelConfig]:
        """Get all models matching a specific tier."""
        return [model for model in self.models.values() if model.tier == tier]
    
    def estimate_cost(self, model_id: str, token_count: int) -> float:
        """Estimate cost for processing given token count with specified model."""
        model = self.get_model(model_id)
        if model:
            return model.cost_per_token * token_count
        return 0.0

class QueryClassifier:
    """Advanced query classification system for routing optimization."""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.classification_cache = {}
        print("🔍 Query Classifier initialized with caching")
    
    async def classify_query(self, query: str, use_cache: bool = True) -> Tuple[QueryComplexity, float, Dict[str, Any]]:
        """
        Classify query complexity with confidence score and metadata.
        
        Args:
            query: User query to classify
            use_cache: Whether to use classification cache
            
        Returns:
            Tuple of (complexity, confidence_score, metadata)
        """
        
        # Check cache first
        if use_cache and query in self.classification_cache:
            cached_result = self.classification_cache[query]
            print(f"📋 Using cached classification for query")
            return cached_result
        
        system_prompt = """You are an expert query classifier that analyzes user prompts and determines their computational complexity.

Classify queries into these categories:

1. **simple**: Direct factual questions, basic lookups, simple calculations
   - Examples: "What is the capital of France?", "Define machine learning", "2+2=?"
   
2. **reasoning**: Logic problems, multi-step inference, analysis requiring thought
   - Examples: "Explain the impact of AI on healthcare", "Compare pros and cons of X vs Y"
   
3. **internet_search**: Current events, recent data, real-time information needs
   - Examples: "Latest news about...", "Current stock price of...", "What happened today in..."
   
4. **complex_analysis**: Deep analysis, research, complex problem-solving, creative tasks
   - Examples: "Design a comprehensive marketing strategy", "Analyze the economic implications of..."

Respond with JSON containing:
- "classification": one of the four categories above
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation of classification
- "estimated_tokens": estimated response length (50-2000)
- "requires_tools": boolean indicating if external tools needed"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",  # Use fast model for classification
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this query: {query}"}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            complexity = QueryComplexity(result["classification"])
            confidence = float(result["confidence"])
            metadata = {
                "reasoning": result.get("reasoning", ""),
                "estimated_tokens": int(result.get("estimated_tokens", 200)),
                "requires_tools": bool(result.get("requires_tools", False))
            }
            
            # Cache the result
            classification_result = (complexity, confidence, metadata)
            if use_cache:
                self.classification_cache[query] = classification_result
            
            print(f"🎯 Classified query as \\{complexity.value} (confidence: \\{confidence:.2f})")
            return classification_result
            
        except Exception as e:
            print(f"⚠️ Classification error: \\{str(e)}, defaulting to REASONING")
            return QueryComplexity.REASONING, 0.5, {"error": str(e), "estimated_tokens": 500}

class RouterAgent:
    """
    Intelligent routing agent that optimizes model selection based on query complexity,
    resource constraints, and performance requirements.
    """
    
    def __init__(self, model_registry: ModelRegistry, resource_budget: ResourceBudget):
        self.model_registry = model_registry
        self.resource_budget = resource_budget
        self.metrics = OptimizationMetrics()
        self.routing_history = []
        print("🚦 Router Agent initialized with intelligent model selection")
    
    def select_optimal_model(self, 
                           complexity: QueryComplexity, 
                           confidence: float,
                           metadata: Dict[str, Any],
                           prefer_speed: bool = False) -> Tuple[str, str]:
        """
        Select optimal model based on complexity, constraints, and preferences.
        
        Returns:
            Tuple of (primary_model_id, fallback_model_id)
        """
        
        estimated_tokens = metadata.get("estimated_tokens", 500)
        
        # Define complexity to model tier mapping
        complexity_mapping = {
            QueryComplexity.SIMPLE: ModelTier.FAST,
            QueryComplexity.REASONING: ModelTier.BALANCED if confidence > 0.8 else ModelTier.PREMIUM,
            QueryComplexity.INTERNET_SEARCH: ModelTier.PREMIUM,
            QueryComplexity.COMPLEX_ANALYSIS: ModelTier.PREMIUM
        }
        
        preferred_tier = complexity_mapping[complexity]
        
        # Adjust for speed preference
        if prefer_speed and preferred_tier != ModelTier.FAST:
            preferred_tier = ModelTier.BALANCED
        
        # Get models for preferred tier
        candidate_models = self.model_registry.get_models_by_tier(preferred_tier)
        
        if not candidate_models:
            # Fallback to any available model
            candidate_models = list(self.model_registry.models.values())
        
        # Filter by budget constraints
        affordable_models = []
        for model in candidate_models:
            estimated_cost = self.model_registry.estimate_cost(
                list(self.model_registry.models.keys())[list(self.model_registry.models.values()).index(model)],
                estimated_tokens
            )
            
            if (self.resource_budget.can_afford(estimated_cost) and 
                estimated_cost <= self.resource_budget.max_cost_per_query):
                affordable_models.append((model, estimated_cost))
        
        if not affordable_models:
            # Emergency fallback to cheapest available model
            print("💰 Budget constraints require cheapest model")
            all_models = [(model, self.model_registry.estimate_cost(
                list(self.model_registry.models.keys())[list(self.model_registry.models.values()).index(model)],
                estimated_tokens
            )) for model in self.model_registry.models.values()]
            
            affordable_models = [min(all_models, key=lambda x: x[1])]
        
        # Select best model from affordable options
        selected_model, estimated_cost = min(affordable_models, key=lambda x: x[1])
        
        # Find model ID
        selected_model_id = None
        for model_id, model_config in self.model_registry.models.items():
            if model_config == selected_model:
                selected_model_id = model_id
                break
        
        # Determine fallback
        fallback_model_id = selected_model.fallback_model or "gpt-4o-mini"
        
        print(f"🎯 Selected model: {selected_model_id} (estimated cost: $\\{estimated_cost:.4f})")
        
        # Log routing decision
        self.routing_history.append({
            "timestamp": datetime.now().isoformat(),
            "complexity": complexity.value,
            "confidence": confidence,
            "selected_model": selected_model_id,
            "fallback_model": fallback_model_id,
            "estimated_cost": estimated_cost,
            "reasoning": f"Tier {preferred_tier.value} for {complexity.value} complexity"
        })
        
        return selected_model_id, fallback_model_id

class CritiqueAgent:
    """
    Quality assurance agent that evaluates response quality and provides feedback
    for continuous optimization of routing decisions.
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.evaluation_history = []
        print("🔍 Critique Agent initialized for quality assurance")
    
    async def evaluate_response(self, 
                              query: str, 
                              response: str, 
                              model_used: str,
                              complexity: QueryComplexity,
                              cost: float,
                              latency: float) -> Dict[str, Any]:
        """
        Comprehensive response evaluation with quality scoring and feedback.
        
        Returns:
            Dictionary with quality metrics and improvement suggestions
        """
        
        evaluation_prompt = f"""You are a quality assurance agent evaluating AI responses. Analyze the following interaction:

**Original Query**: \\{query}
**AI Response**: \\{response}
**Model Used**: \\{model_used}
**Expected Complexity**: \\{complexity.value}
**Cost**: $\\{cost:.4f}
**Latency**: \\{latency:.0f}ms

Evaluate on these dimensions (score 1-10):
1. **Accuracy**: Factual correctness and reliability
2. **Completeness**: Addresses all aspects of the query
3. **Clarity**: Clear, well-structured, easy to understand
4. **Relevance**: Directly addresses the question asked
5. **Efficiency**: Appropriate response for the complexity level

Also assess:
- **Cost Effectiveness**: Was the model selection appropriate for this query?
- **Speed Appropriateness**: Was the latency acceptable for this type of query?
- **Routing Quality**: Should this query have been routed differently?

Respond with JSON containing:
{{
    "scores": {{
        "accuracy": int,
        "completeness": int, 
        "clarity": int,
        "relevance": int,
        "efficiency": int
    }},
    "overall_quality": float (1-10),
    "cost_effectiveness": float (1-10),
    "speed_rating": float (1-10),
    "routing_assessment": "optimal|suboptimal|poor",
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "recommended_model": "model_name or null",
    "feedback_summary": "brief summary of evaluation"
}}"""
        
        try:
            response_eval = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1
            )
            
            evaluation = json.loads(response_eval.choices[0].message.content)
            
            # Add metadata
            evaluation["evaluation_timestamp"] = datetime.now().isoformat()
            evaluation["model_evaluated"] = model_used
            evaluation["original_complexity"] = complexity.value
            evaluation["actual_cost"] = cost
            evaluation["actual_latency"] = latency
            
            # Store evaluation
            self.evaluation_history.append(evaluation)
            
            print(f"📊 Response evaluation: Quality \\{evaluation['overall_quality']:.1f}/10, "
                  f"Cost Effectiveness \\{evaluation['cost_effectiveness']:.1f}/10")
            
            return evaluation
            
        except Exception as e:
            print(f"⚠️ Evaluation error: \\{str(e)}")
            return {
                "overall_quality": 5.0,
                "cost_effectiveness": 5.0,
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def get_routing_insights(self) -> Dict[str, Any]:
        """Generate insights for improving routing decisions based on evaluation history."""
        
        if not self.evaluation_history:
            return {"message": "No evaluations available yet"}
        
        # Analyze routing effectiveness
        routing_assessments = [eval["routing_assessment"] for eval in self.evaluation_history 
                             if "routing_assessment" in eval]
        
        optimal_count = routing_assessments.count("optimal")
        total_assessments = len(routing_assessments)
        
        # Average scores by model
        model_performance = {}
        for evaluation in self.evaluation_history:
            model = evaluation.get("model_evaluated", "unknown")
            quality = evaluation.get("overall_quality", 0)
            cost_eff = evaluation.get("cost_effectiveness", 0)
            
            if model not in model_performance:
                model_performance[model] = {"quality": [], "cost_effectiveness": []}
            
            model_performance[model]["quality"].append(quality)
            model_performance[model]["cost_effectiveness"].append(cost_eff)
        
        # Calculate averages
        model_averages = {}
        for model, metrics in model_performance.items():
            model_averages[model] = {
                "avg_quality": statistics.mean(metrics["quality"]),
                "avg_cost_effectiveness": statistics.mean(metrics["cost_effectiveness"]),
                "evaluation_count": len(metrics["quality"])
            }
        
        return {
            "routing_success_rate": optimal_count / max(total_assessments, 1),
            "total_evaluations": len(self.evaluation_history),
            "model_performance": model_averages,
            "insights": self._generate_routing_insights(model_averages, routing_assessments)
        }
    
    def _generate_routing_insights(self, model_averages: Dict, assessments: List[str]) -> List[str]:
        """Generate actionable insights for routing optimization."""
        insights = []
        
        if len(model_averages) > 1:
            best_quality = max(model_averages.items(), key=lambda x: x[1]["avg_quality"])
            best_cost = max(model_averages.items(), key=lambda x: x[1]["avg_cost_effectiveness"])
            
            insights.append(f"Highest quality responses: {best_quality[0]} ({best_quality[1]['avg_quality']:.1f}/10)")
            insights.append(f"Most cost-effective: {best_cost[0]} ({best_cost[1]['avg_cost_effectiveness']:.1f}/10)")
        
        suboptimal_rate = assessments.count("suboptimal") / max(len(assessments), 1)
        if suboptimal_rate > 0.2:
            insights.append(f"High suboptimal routing rate ({suboptimal_rate:.1%}) - consider adjusting classification thresholds")
        
        return insights

class ResourceAwareOptimizer:
    """
    Main optimization system coordinating all components for intelligent resource management.
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.resource_budget = ResourceBudget(
            max_cost_per_query=2.0,
            max_latency_ms=30000,
            daily_budget=50.0
        )
        
        self.query_classifier = QueryClassifier(self.openai_client)
        self.router_agent = RouterAgent(self.model_registry, self.resource_budget)
        self.critique_agent = CritiqueAgent(self.openai_client)
        
        # Performance tracking
        self.session_start = datetime.now()
        
        print("🚀 Resource-Aware Optimizer initialized successfully")
        print(f"💰 Daily budget: $\\{self.resource_budget.daily_budget}")
        print(f"⏱️ Max latency: \\{self.resource_budget.max_latency_ms}ms")
    
    async def process_query(self, query: str, prefer_speed: bool = False) -> Dict[str, Any]:
        """
        Process query with full resource-aware optimization pipeline.
        
        Args:
            query: User query to process
            prefer_speed: Whether to prioritize speed over quality
            
        Returns:
            Comprehensive result including response, metrics, and analysis
        """
        
        start_time = time.time()
        
        print(f"\\n🔄 Processing query: \\{query[:100]}\\{'...' if len(query) > 100 else ''}")
        
        try:
            # Step 1: Classify query complexity
            complexity, confidence, metadata = await self.query_classifier.classify_query(query)
            
            # Step 2: Select optimal model
            primary_model, fallback_model = self.router_agent.select_optimal_model(
                complexity, confidence, metadata, prefer_speed
            )
            
            # Step 3: Generate response with fallback handling
            response, actual_model, actual_cost = await self._generate_response_with_fallback(
                query, primary_model, fallback_model, metadata
            )
            
            # Step 4: Calculate performance metrics
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Step 5: Evaluate response quality
            evaluation = await self.critique_agent.evaluate_response(
                query, response, actual_model, complexity, actual_cost, latency
            )
            
            # Step 6: Update budget and metrics
            self.resource_budget.update_spend(actual_cost)
            self.router_agent.metrics.add_query_result(
                actual_cost, latency, evaluation.get("overall_quality", 0), actual_model
            )
            
            # Prepare comprehensive result
            result = {
                "query": query,
                "response": response,
                "model_used": actual_model,
                "complexity": complexity.value,
                "confidence": confidence,
                "cost": actual_cost,
                "latency_ms": latency,
                "evaluation": evaluation,
                "budget_remaining": self.resource_budget.daily_budget - self.resource_budget.current_daily_spend,
                "optimization_successful": evaluation.get("routing_assessment") == "optimal"
            }
            
            print(f"✅ Query processed successfully in \\{latency:.0f}ms for $\\{actual_cost:.4f}")
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: \\{str(e)}")
            return {
                "query": query,
                "error": str(e),
                "fallback_response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "cost": 0.0,
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    async def _generate_response_with_fallback(self, 
                                             query: str, 
                                             primary_model: str, 
                                             fallback_model: str,
                                             metadata: Dict[str, Any]) -> Tuple[str, str, float]:
        """Generate response with automatic fallback on failure."""
        
        models_to_try = [primary_model]
        if fallback_model and fallback_model != primary_model:
            models_to_try.append(fallback_model)
        
        last_error = None
        
        for model_id in models_to_try:
            try:
                print(f"🤖 Attempting generation with \\{model_id}")
                
                model_config = self.model_registry.get_model(model_id)
                estimated_tokens = metadata.get("estimated_tokens", 500)
                estimated_cost = self.model_registry.estimate_cost(model_id, estimated_tokens)
                
                # Check budget before making API call
                if not self.resource_budget.can_afford(estimated_cost):
                    print(f"💰 Insufficient budget for \\{model_id}")
                    continue
                
                # Make API call (simplified - would use appropriate client based on model)
                response = await self._make_model_api_call(model_id, query, model_config)
                
                print(f"✅ Successfully generated response with \\{model_id}")
                return response, model_id, estimated_cost
                
            except Exception as e:
                print(f"⚠️ Model \\{model_id} failed: \\{str(e)}")
                last_error = e
                
                # Track fallback usage
                if model_id != primary_model:
                    self.router_agent.metrics.fallback_count += 1
                
                continue
        
        # All models failed
        raise Exception(f"All models failed. Last error: \\{str(last_error)}")
    
    async def _make_model_api_call(self, model_id: str, query: str, model_config: ModelConfig) -> str:
        """Make API call to specified model (simplified implementation)."""
        
        # This would be replaced with actual model-specific API calls
        # For demonstration, using OpenAI API with different models
        
        if "gpt" in model_id:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model_config.name,
                messages=[{"role": "user", "content": query}],
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Placeholder for other model APIs (Gemini, etc.)
            return f"Response from \\{model_id}: \\{query[:50]}... [Generated response would appear here]"
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive system performance analytics."""
        
        uptime = datetime.now() - self.session_start
        performance_summary = self.router_agent.metrics.get_performance_summary()
        routing_insights = self.critique_agent.get_routing_insights()
        
        budget_utilization = (self.resource_budget.current_daily_spend / 
                            self.resource_budget.daily_budget) * 100
        
        return {
            "system_uptime": str(uptime),
            "budget_status": {
                "daily_budget": self.resource_budget.daily_budget,
                "spent": self.resource_budget.current_daily_spend,
                "remaining": self.resource_budget.daily_budget - self.resource_budget.current_daily_spend,
                "utilization_percentage": budget_utilization
            },
            "performance_metrics": performance_summary,
            "routing_insights": routing_insights,
            "system_health": {
                "healthy": budget_utilization < 90 and performance_summary["fallback_rate"] < 0.1,
                "warnings": self._generate_health_warnings(budget_utilization, performance_summary)
            }
        }
    
    def _generate_health_warnings(self, budget_util: float, perf_summary: Dict[str, Any]) -> List[str]:
        """Generate system health warnings based on current metrics."""
        warnings = []
        
        if budget_util > 90:
            warnings.append("High budget utilization - consider implementing stricter cost controls")
        
        if perf_summary["fallback_rate"] > 0.2:
            warnings.append("High fallback rate - primary models may be unreliable")
        
        if perf_summary["average_latency_ms"] > 10000:
            warnings.append("High average latency - consider faster model selection")
        
        avg_quality = perf_summary.get("average_quality", 0)
        if avg_quality < 7.0:
            warnings.append("Low average quality scores - routing strategy may need adjustment")
        
        return warnings

# Demonstration and Testing Functions
async def demonstrate_resource_optimization():
    """Comprehensive demonstration of resource-aware optimization system."""
    
    print("🎯 RESOURCE-AWARE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Initialize optimizer
    optimizer = ResourceAwareOptimizer()
    
    # Test queries of varying complexity
    test_queries = [
        ("What is the capital of France?", False),  # Simple
        ("Explain the implications of quantum computing on modern cryptography", False),  # Complex
        ("What are the latest developments in AI research this week?", True),  # Current info, prefer speed
        ("Design a comprehensive marketing strategy for a new sustainable fashion brand targeting Gen Z consumers", False),  # Complex analysis
        ("Calculate the compound interest on $1000 at 5% for 3 years", False)  # Simple calculation
    ]
    
    print(f"\\n🧪 Testing \\{len(test_queries)} queries with different complexity levels\\n")
    
    results = []
    for i, (query, prefer_speed) in enumerate(test_queries, 1):
        print(f"\\n{'='*20} Query {i} {'='*20}")
        print(f"Speed Preference: {'HIGH' if prefer_speed else 'BALANCED'}")
        
        result = await optimizer.process_query(query, prefer_speed=prefer_speed)
        results.append(result)
        
        # Display key metrics
        if "error" not in result:
            print(f"🎯 Complexity: \\{result['complexity']}")
            print(f"🤖 Model: \\{result['model_used']}")
            print(f"💰 Cost: $\\{result['cost']:.4f}")
            print(f"⏱️ Latency: \\{result['latency_ms']:.0f}ms")
            print(f"⭐ Quality: \\{result['evaluation'].get('overall_quality', 'N/A')}/10")
            print(f"💡 Optimization: \\{'✅ Optimal' if result['optimization_successful'] else '⚠️ Suboptimal'}")
        else:
            print(f"❌ Error: \\{result['error']}")
        
        # Brief pause between queries
        await asyncio.sleep(1)
    
    # Generate final analytics
    print(f"\\n\\n📊 SYSTEM ANALYTICS")
    print("="*40)
    
    analytics = optimizer.get_system_analytics()
    
    print(f"💰 Budget Status:")
    budget = analytics["budget_status"]
    print(f"   Daily Budget: $\\{budget['daily_budget']:.2f}")
    print(f"   Spent: $\\{budget['spent']:.4f}")
    print(f"   Remaining: $\\{budget['remaining']:.4f}")
    print(f"   Utilization: \\{budget['utilization_percentage']:.1f}%")
    
    print(f"\\n📈 Performance Metrics:")
    perf = analytics["performance_metrics"]
    print(f"   Total Queries: \\{perf['total_queries']}")
    print(f"   Average Cost: $\\{perf['average_cost_per_query']:.4f}")
    print(f"   Average Latency: \\{perf['average_latency_ms']:.0f}ms")
    print(f"   Average Quality: \\{perf.get('average_quality', 0):.1f}/10")
    print(f"   Fallback Rate: \\{perf['fallback_rate']:.1%}")
    
    print(f"\\n🎯 Routing Insights:")
    insights = analytics["routing_insights"]
    print(f"   Routing Success: \\{insights.get('routing_success_rate', 0):.1%}")
    print(f"   Total Evaluations: \\{insights.get('total_evaluations', 0)}")
    
    health = analytics["system_health"]
    print(f"\\n🏥 System Health: {'✅ Healthy' if health['healthy'] else '⚠️ Needs Attention'}")
    if health['warnings']:
        for warning in health['warnings']:
            print(f"   ⚠️ \\{warning}")
    
    print("\\n✅ Resource-Aware Optimization Demonstration Complete!")
    print(f"Successfully processed \\{len([r for r in results if 'error' not in r])} queries")
    print(f"System optimized for cost efficiency and performance balance")

# Example usage
if __name__ == "__main__":
    asyncio.run(demonstrate_resource_optimization())`,

    practicalApplications: [
      '💰 Cost-Optimized LLM Usage: Dynamic selection between expensive high-capability models for complex reasoning and affordable fast models for simple queries based on budget constraints and task complexity',
      '⚡ Latency-Sensitive Operations: Real-time systems choosing faster reasoning paths over comprehensive analysis to ensure timely responses in trading, customer service, and interactive applications',
      '🔋 Energy-Efficient Edge Deployment: Agents on mobile devices or IoT systems optimizing processing to conserve battery life while maintaining acceptable performance levels',
      '🔄 Service Reliability Through Fallbacks: Automatic switching to backup models when primary choices are unavailable due to rate limits, ensuring service continuity and graceful degradation',
      '📊 Data Usage Management: Smart bandwidth optimization choosing summarized data retrieval over full datasets, compressed formats, or cached results to minimize transfer costs',
      '⚖️ Adaptive Task Allocation: Multi-agent systems where agents self-assign tasks based on current computational load, available time, and resource capacity for optimal distribution',
      '🏢 Enterprise Cost Management: Large-scale deployments with sophisticated budget tracking, departmental cost allocation, and automatic scaling based on usage patterns and financial constraints',
      '🎯 Quality-Cost Balance Optimization: Dynamic quality adjustment based on use case importance - high precision for critical decisions, standard quality for routine operations'
    ],

    nextSteps: [
      'Start with basic router agent implementation classifying query complexity and routing to appropriate models based on simple metrics like query length and keyword analysis',
      'Implement comprehensive cost tracking and budget management systems with real-time monitoring, alerts, and automatic scaling based on spend rate and performance metrics',
      'Design fallback mechanisms ensuring graceful degradation when preferred models are unavailable, with multiple fallback tiers and automatic recovery procedures',
      'Build critique agent systems for continuous quality assessment, feedback loops, and routing optimization based on performance evaluation and user satisfaction',
      'Integrate with multiple model providers (OpenAI, Google, Anthropic) through unified interfaces like OpenRouter for automated model selection and cost optimization',
      'Establish comprehensive monitoring and analytics systems tracking resource utilization, performance metrics, cost efficiency, and quality trade-offs',
      'Implement advanced optimization techniques including contextual pruning, proactive resource prediction, and learned resource allocation policies',
      'Design production-ready architectures with horizontal scaling, load balancing, caching strategies, and enterprise-grade security for large-scale deployments'
    ]
  },

  sections: [
    {
      title: 'Dynamic Model Selection and Router Agent Architecture',
      content: `Dynamic model selection forms the cornerstone of resource-aware optimization, enabling intelligent agents to automatically choose the most appropriate computational resources based on task complexity, performance requirements, and resource constraints through sophisticated router agent architectures.

**Query Complexity Classification Systems**

**Multi-Dimensional Complexity Analysis**
Advanced query classification goes beyond simple metrics to understand true computational requirements:
- **Semantic Complexity Analysis**: Using LLMs to understand the depth of reasoning required, factual vs. analytical needs, and conceptual difficulty
- **Computational Requirement Prediction**: Estimating token consumption, processing time, and memory requirements based on query characteristics
- **Context Dependency Assessment**: Analyzing whether queries require external data, multi-step reasoning, or domain-specific knowledge
- **Confidence Scoring**: Providing uncertainty measures for classification decisions to inform routing strategies

**Hierarchical Classification Framework**
Structured approach to complexity categorization:
- **Simple Queries**: Direct factual questions, basic lookups, simple calculations requiring minimal reasoning
- **Reasoning Tasks**: Logic problems, multi-step inference, comparative analysis requiring structured thought processes
- **Research and Analysis**: Complex investigations, synthesis from multiple sources, strategic thinking requiring advanced reasoning
- **Creative and Generative**: Content creation, design tasks, creative problem-solving requiring imagination and original thinking

**Real-Time Classification Optimization**
Ensuring classification efficiency and accuracy:
- **Classification Caching**: Storing classification results for similar queries to reduce redundant analysis
- **Incremental Learning**: Improving classification accuracy based on routing outcomes and performance feedback
- **Multi-Model Classification**: Using ensemble approaches with multiple classification models for improved accuracy
- **Contextual Classification**: Considering user history, session context, and domain-specific factors in classification decisions

**Router Agent Decision Framework**

**Multi-Criteria Decision Making**
Sophisticated decision algorithms balancing multiple optimization objectives:
- **Cost-Performance Trade-offs**: Mathematical optimization functions balancing response quality against computational costs
- **Latency Requirements**: Time-sensitive routing decisions prioritizing speed when immediate responses are required
- **Quality Thresholds**: Minimum acceptable quality levels ensuring appropriate model selection for task criticality
- **Resource Availability**: Real-time assessment of model availability, rate limits, and computational resource status

**Advanced Routing Strategies**
Intelligent routing patterns for optimal resource utilization:
\`\`\`python
class AdvancedRouter:
    def __init__(self):
        self.routing_strategies = {
            'cost_optimized': self._cost_optimized_routing,
            'quality_first': self._quality_first_routing, 
            'balanced': self._balanced_routing,
            'speed_priority': self._speed_priority_routing
        }
        
        self.model_capabilities = {
            'simple_tasks': ['gpt-3.5-turbo', 'claude-instant', 'gemini-flash'],
            'reasoning_tasks': ['gpt-4', 'claude-3-sonnet', 'gemini-pro'],
            'complex_analysis': ['gpt-4-turbo', 'claude-3-opus', 'gemini-ultra'],
            'specialized_domains': ['gpt-4-turbo', 'claude-3-opus']
        }
        
        self.performance_history = {}
        
    def route_query(self, query_analysis, constraints, strategy='balanced'):
        """Route query using specified strategy and constraints."""
        
        # Extract key factors
        complexity = query_analysis['complexity']
        confidence = query_analysis['confidence']
        estimated_tokens = query_analysis['estimated_tokens']
        domain = query_analysis.get('domain', 'general')
        
        # Apply routing strategy
        routing_func = self.routing_strategies[strategy]
        model_selection = routing_func(complexity, confidence, estimated_tokens, constraints)
        
        # Validate selection against constraints
        validated_selection = self._validate_selection(model_selection, constraints)
        
        return validated_selection
        
    def _cost_optimized_routing(self, complexity, confidence, tokens, constraints):
        """Route prioritizing cost efficiency while maintaining acceptable quality."""
        budget_per_query = constraints.get('max_cost_per_query', 1.0)
        
        # Get cost-effective models for complexity level
        if complexity == 'simple' and confidence > 0.8:
            candidates = self.model_capabilities['simple_tasks']
        elif complexity in ['reasoning', 'analysis']:
            # Use cheaper reasoning models if confidence is high
            candidates = (self.model_capabilities['simple_tasks'] if confidence > 0.9 
                         else self.model_capabilities['reasoning_tasks'])
        else:
            candidates = self.model_capabilities['reasoning_tasks']
            
        # Select cheapest model within budget
        for model in candidates:
            estimated_cost = self._estimate_cost(model, tokens)
            if estimated_cost <= budget_per_query:
                return {'primary': model, 'estimated_cost': estimated_cost}
                
        # Fallback to cheapest available
        return {'primary': 'gpt-3.5-turbo', 'estimated_cost': 0.002 * tokens}
        
    def _quality_first_routing(self, complexity, confidence, tokens, constraints):
        """Route prioritizing response quality regardless of cost."""
        quality_threshold = constraints.get('min_quality_score', 8.0)
        
        # Map complexity to highest quality models
        quality_mapping = {
            'simple': 'gpt-4' if quality_threshold > 8.5 else 'gpt-3.5-turbo',
            'reasoning': 'gpt-4-turbo',
            'analysis': 'claude-3-opus',
            'complex_analysis': 'gpt-4-turbo'
        }
        
        selected_model = quality_mapping.get(complexity, 'gpt-4')
        return {
            'primary': selected_model,
            'estimated_cost': self._estimate_cost(selected_model, tokens)
        }
\`\`\`

**Dynamic Resource Monitoring**
Real-time resource tracking for informed routing decisions:
- **Model Availability Monitoring**: Real-time status of different models including rate limits, outages, and performance degradation
- **Cost Tracking**: Live budget monitoring with predictive spending analysis and automatic budget protection
- **Performance Metrics**: Response time, quality scores, and success rates for different model combinations
- **Capacity Planning**: Predictive analysis of resource needs based on usage patterns and demand forecasting

**Fallback and Recovery Mechanisms**

**Intelligent Fallback Strategies**
Comprehensive fallback systems ensuring service continuity:
- **Tiered Fallback Systems**: Multiple fallback levels from premium models to budget options to emergency basic models
- **Context-Aware Fallback**: Choosing fallback models that maintain capability alignment with original task requirements
- **Quality-Preserving Degradation**: Selecting fallback options that minimize quality impact while staying within constraints
- **Cross-Provider Fallback**: Routing to alternative providers when primary services are unavailable

**Recovery and Optimization Learning**
Learning from failures and suboptimal routing decisions:
- **Failure Pattern Analysis**: Identifying common failure modes and developing preemptive routing strategies
- **Performance Feedback Integration**: Using quality assessments to improve future routing decisions
- **Cost-Quality Correlation Analysis**: Understanding relationship between model costs and output quality for different task types
- **Adaptive Threshold Management**: Dynamically adjusting routing thresholds based on performance history and current conditions

This comprehensive approach to dynamic model selection ensures that resource-aware systems can optimize performance across multiple dimensions while adapting to changing conditions and learning from experience to improve future decisions.`
    },
    {
      title: 'Cost Management and Budget Optimization Strategies',
      content: `Effective cost management in resource-aware systems requires sophisticated budget tracking, predictive cost modeling, and intelligent optimization strategies that balance financial constraints with performance requirements across diverse operational scenarios.

**Comprehensive Budget Management Framework**

**Multi-Level Budget Architecture**
Hierarchical budget management supporting complex organizational structures:
- **Global Budget Limits**: Organization-wide spending caps with automatic enforcement and alert systems
- **Department/Team Budgets**: Granular allocation enabling departmental cost control and accountability tracking
- **Project-Specific Budgets**: Time-bound project budgets with milestone-based spending and resource allocation
- **User-Level Quotas**: Individual user limits preventing abuse while enabling productivity

**Real-Time Budget Monitoring**
Advanced tracking systems providing immediate visibility into resource consumption:
- **Live Spend Tracking**: Real-time cost accumulation with immediate updates and threshold monitoring
- **Predictive Spend Analysis**: Forecasting future costs based on current usage patterns and trend analysis
- **Budget Burn Rate Monitoring**: Tracking spending velocity with alerts for unsustainable consumption rates
- **Cost Attribution**: Detailed breakdown of costs by user, department, project, and model usage

**Dynamic Budget Allocation**
Intelligent budget distribution adapting to changing needs:
- **Demand-Based Reallocation**: Shifting budget allocation based on actual usage patterns and priority changes
- **Seasonal Adjustment**: Automatic budget modifications accounting for predictable usage cycles
- **Emergency Budget Reserves**: Reserved funds for critical operations and unexpected high-priority needs
- **Cross-Department Sharing**: Flexible budget sharing mechanisms for collaborative projects and resource optimization

**Advanced Cost Optimization Techniques**

**Predictive Cost Modeling**
Sophisticated forecasting for proactive cost management:
\`\`\`python
class PredictiveCostModel:
    def __init__(self):
        self.usage_history = []
        self.cost_models = {}
        self.seasonal_patterns = {}
        
    def predict_query_cost(self, query_analysis, model_selection):
        """Predict total cost including primary and potential fallback costs."""
        
        base_cost = self._calculate_base_cost(query_analysis, model_selection)
        
        # Factor in probability of fallback usage
        fallback_probability = self._estimate_fallback_probability(
            model_selection['primary'], query_analysis['complexity']
        )
        
        fallback_cost = 0
        if fallback_probability > 0:
            fallback_models = model_selection.get('fallbacks', [])
            fallback_cost = sum(
                self._calculate_base_cost(query_analysis, {'primary': fb_model}) * 
                (fallback_probability / len(fallback_models))
                for fb_model in fallback_models
            )
        
        # Add overhead costs (monitoring, logging, etc.)
        overhead_cost = base_cost * 0.05  # 5% overhead
        
        total_predicted_cost = base_cost + fallback_cost + overhead_cost
        
        return {
            'base_cost': base_cost,
            'fallback_cost': fallback_cost,
            'overhead_cost': overhead_cost,
            'total_cost': total_predicted_cost,
            'confidence': self._get_prediction_confidence(query_analysis)
        }
        
    def predict_daily_spend(self, current_usage_rate, time_remaining_hours):
        """Predict total daily spend based on current usage patterns."""
        
        # Calculate current hourly rate
        if not self.usage_history:
            return {'prediction': 0, 'confidence': 0.0}
            
        recent_usage = self.usage_history[-24:]  # Last 24 hours
        hourly_rates = [hour['cost'] for hour in recent_usage]
        
        if not hourly_rates:
            return {'prediction': 0, 'confidence': 0.0}
            
        # Apply time-of-day patterns
        current_hour = datetime.now().hour
        hourly_multiplier = self.seasonal_patterns.get('hourly', {}).get(current_hour, 1.0)
        
        base_hourly_rate = statistics.mean(hourly_rates) * hourly_multiplier
        projected_spend = base_hourly_rate * time_remaining_hours
        
        return {
            'prediction': projected_spend,
            'confidence': min(0.9, len(hourly_rates) / 24.0),
            'hourly_rate': base_hourly_rate,
            'time_remaining': time_remaining_hours
        }
        
    def optimize_model_selection_for_budget(self, query_analysis, available_budget, quality_threshold):
        """Select optimal model given budget constraints and quality requirements."""
        
        candidate_models = self._get_candidate_models(query_analysis['complexity'])
        
        viable_options = []
        for model in candidate_models:
            cost_prediction = self.predict_query_cost(query_analysis, {'primary': model})
            
            if cost_prediction['total_cost'] <= available_budget:
                expected_quality = self._estimate_quality(model, query_analysis)
                
                if expected_quality >= quality_threshold:
                    viable_options.append({
                        'model': model,
                        'cost': cost_prediction['total_cost'],
                        'quality': expected_quality,
                        'efficiency': expected_quality / cost_prediction['total_cost']
                    })
        
        if not viable_options:
            return None
            
        # Select highest efficiency option
        best_option = max(viable_options, key=lambda x: x['efficiency'])
        return best_option
\`\`\`

**Cost-Quality Optimization**
Balancing financial constraints with output quality requirements:
- **Quality-Cost Curves**: Mathematical models relating model costs to expected output quality
- **Minimum Viable Quality**: Defining acceptable quality thresholds for different use cases and contexts
- **Cost-Benefit Analysis**: Evaluating whether increased spending yields proportional quality improvements
- **Dynamic Quality Adjustment**: Automatically adjusting quality requirements based on budget availability

**Intelligent Cost Control Mechanisms**

**Proactive Budget Protection**
Preventing budget overruns through intelligent safeguards:
- **Predictive Budget Alerts**: Early warning systems based on spending trajectory analysis
- **Automatic Budget Enforcement**: Hard stops preventing spending beyond allocated budgets
- **Graceful Degradation Triggers**: Automatic switching to lower-cost models when budget limits approach
- **Emergency Budget Protocols**: Reserve budget allocation for critical operations

**Cost Optimization Algorithms**
Mathematical approaches to minimize costs while maintaining performance:
- **Linear Programming**: Optimizing model selection across multiple constraints simultaneously
- **Dynamic Programming**: Finding optimal routing strategies for complex multi-step workflows
- **Genetic Algorithms**: Evolving optimal cost-performance configurations through iterative improvement
- **Reinforcement Learning**: Learning optimal cost management strategies through trial and performance feedback

**Advanced Pricing and Procurement Strategies**

**Multi-Provider Cost Arbitrage**
Leveraging multiple AI service providers for optimal pricing:
- **Real-Time Price Comparison**: Comparing costs across providers for equivalent capabilities
- **Bulk Purchasing Optimization**: Negotiating volume discounts and reserved capacity pricing
- **Geographic Cost Arbitrage**: Routing requests to lower-cost regions when latency permits
- **Peak/Off-Peak Optimization**: Scheduling non-urgent requests during lower-cost time periods

**Contract and Pricing Optimization**
Strategic approaches to AI service procurement:
- **Reserved Capacity Planning**: Balancing committed spend discounts against usage flexibility
- **Spot Instance Utilization**: Using variable pricing models for fault-tolerant workloads
- **Custom Pricing Negotiations**: Leveraging high usage volumes for better pricing terms
- **Multi-Cloud Strategies**: Avoiding vendor lock-in while optimizing for cost and performance

**Cost Analytics and Reporting**

**Comprehensive Cost Visibility**
Detailed analysis and reporting of resource consumption patterns:
- **Cost Attribution Analytics**: Understanding cost drivers by user, department, project, and use case
- **Trend Analysis**: Identifying spending patterns, seasonal variations, and growth trajectories
- **Efficiency Metrics**: Measuring cost per successful query, cost per quality point, and ROI analysis
- **Benchmarking**: Comparing costs against industry standards and organizational targets

**Optimization Recommendations**
Actionable insights for cost reduction and efficiency improvement:
- **Usage Pattern Analysis**: Identifying opportunities for batch processing and off-peak scheduling
- **Model Efficiency Assessment**: Recommending model changes based on cost-quality analysis
- **Workflow Optimization**: Suggesting process improvements to reduce overall computational requirements
- **Budget Reallocation**: Recommending budget redistribution based on actual usage patterns and priorities

This comprehensive cost management framework ensures that resource-aware systems can operate within financial constraints while maintaining optimal performance and providing clear visibility into resource utilization and optimization opportunities.`
    },
    {
      title: 'Quality Assessment and Critique Agent Systems',
      content: `Quality assessment and critique agent systems provide essential feedback loops for resource-aware optimization, enabling continuous improvement of routing decisions, model selection strategies, and overall system performance through systematic evaluation and intelligent feedback mechanisms.

**Comprehensive Quality Evaluation Framework**

**Multi-Dimensional Quality Metrics**
Sophisticated assessment covering multiple aspects of response quality:
- **Accuracy and Factual Correctness**: Verification of factual claims, numerical calculations, and logical consistency
- **Completeness and Coverage**: Assessment of whether responses address all aspects of the query comprehensively
- **Clarity and Communication**: Evaluation of response structure, readability, and effective communication of concepts
- **Relevance and Focus**: Measurement of how directly responses address the specific question or requirement
- **Depth and Insight**: Analysis of analytical depth, creative thinking, and value-added insights

**Context-Aware Quality Assessment**
Tailoring evaluation criteria based on query characteristics and usage context:
- **Domain-Specific Evaluation**: Applying specialized quality criteria for technical, creative, or analytical domains
- **User Intent Recognition**: Adjusting quality expectations based on identified user needs and objectives
- **Criticality-Based Assessment**: Higher standards for high-stakes queries versus casual information requests
- **Cultural and Linguistic Sensitivity**: Ensuring quality evaluation accounts for diverse cultural contexts and languages

**Automated Quality Scoring Systems**
Scalable automated assessment enabling real-time quality monitoring:
\`\`\`python
class AdvancedQualityAssessment:
    def __init__(self):
        self.evaluation_models = {
            'factual_accuracy': self._assess_factual_accuracy,
            'completeness': self._assess_completeness,
            'clarity': self._assess_clarity,
            'relevance': self._assess_relevance,
            'creativity': self._assess_creativity
        }
        
        self.domain_specific_evaluators = {
            'technical': TechnicalQualityEvaluator(),
            'creative': CreativeQualityEvaluator(),
            'analytical': AnalyticalQualityEvaluator(),
            'conversational': ConversationalQualityEvaluator()
        }
        
        self.quality_benchmarks = {}
        self.evaluation_history = []
        
    async def comprehensive_quality_assessment(self, query, response, context):
        """Perform comprehensive quality assessment across multiple dimensions."""
        
        # Determine assessment context
        domain = self._identify_domain(query, context)
        criticality = self._assess_criticality(query, context)
        user_expectations = self._infer_user_expectations(query, context)
        
        # Core quality dimensions
        quality_scores = {}
        for dimension, evaluator in self.evaluation_models.items():
            score = await evaluator(query, response, context)
            quality_scores[dimension] = score
            
        # Domain-specific evaluation
        domain_evaluation = None
        if domain in self.domain_specific_evaluators:
            domain_evaluator = self.domain_specific_evaluators[domain]
            domain_evaluation = await domain_evaluator.evaluate(query, response, context)
            
        # Calculate weighted overall score
        weights = self._get_dimension_weights(domain, criticality, user_expectations)
        overall_score = sum(score * weights.get(dim, 1.0) 
                           for dim, score in quality_scores.items()) / sum(weights.values())
        
        # Generate detailed assessment
        assessment = {
            'overall_score': overall_score,
            'dimension_scores': quality_scores,
            'domain_evaluation': domain_evaluation,
            'context': {
                'domain': domain,
                'criticality': criticality,
                'user_expectations': user_expectations
            },
            'benchmarks': self._get_quality_benchmarks(domain, criticality),
            'improvement_suggestions': self._generate_improvement_suggestions(
                quality_scores, domain_evaluation, overall_score
            )
        }
        
        # Store evaluation for learning
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_hash': hash(query),
            'assessment': assessment
        })
        
        return assessment
        
    def _assess_factual_accuracy(self, query, response, context):
        """Evaluate factual accuracy of response content."""
        
        # Extract factual claims from response
        factual_claims = self._extract_factual_claims(response)
        
        if not factual_claims:
            return 8.0  # No factual claims to verify
            
        # Verify claims against knowledge base
        verified_claims = 0
        total_claims = len(factual_claims)
        
        for claim in factual_claims:
            if self._verify_factual_claim(claim, context):
                verified_claims += 1
                
        accuracy_rate = verified_claims / total_claims
        
        # Convert to 1-10 scale
        return 1 + (accuracy_rate * 9)
        
    def _assess_completeness(self, query, response, context):
        """Evaluate whether response comprehensively addresses the query."""
        
        # Parse query to identify required components
        query_components = self._parse_query_requirements(query)
        
        # Analyze response coverage
        covered_components = 0
        for component in query_components:
            if self._response_addresses_component(response, component):
                covered_components += 1
                
        completeness_rate = covered_components / max(len(query_components), 1)
        
        # Adjust for depth and detail
        depth_score = self._assess_response_depth(response, query_components)
        
        combined_score = (completeness_rate * 0.7) + (depth_score * 0.3)
        
        return 1 + (combined_score * 9)
\`\`\`

**Critique Agent Architecture**

**Intelligent Feedback Generation**
Advanced critique systems providing actionable improvement recommendations:
- **Constructive Criticism**: Identifying specific areas for improvement with actionable suggestions
- **Strength Recognition**: Acknowledging response strengths to reinforce positive patterns
- **Context-Sensitive Feedback**: Tailoring critique based on query type, user level, and intended use
- **Progressive Improvement**: Tracking improvement over time and adjusting feedback accordingly

**Multi-Level Critique Systems**
Hierarchical critique providing different levels of analysis:
- **Real-Time Basic Critique**: Fast, automated assessment for immediate routing feedback
- **Comprehensive Periodic Review**: Detailed analysis of response patterns and quality trends
- **Expert Human Review**: Human expert validation of critical or ambiguous assessments
- **Peer Review Systems**: Cross-validation between multiple critique agents for consensus building

**Learning and Adaptation Mechanisms**

**Continuous Improvement Loops**
Systems that learn from critique feedback to improve future performance:
- **Routing Decision Optimization**: Adjusting model selection based on quality assessment outcomes
- **Threshold Calibration**: Fine-tuning quality thresholds based on actual performance data
- **Model Performance Tracking**: Building detailed performance profiles for different models and use cases
- **User Satisfaction Correlation**: Linking quality scores with actual user satisfaction and feedback

**Adaptive Assessment Criteria**
Quality standards that evolve based on system capabilities and user expectations:
- **Dynamic Benchmark Adjustment**: Updating quality standards as model capabilities improve
- **Context-Dependent Standards**: Adjusting expectations based on task difficulty and resource constraints
- **User Preference Learning**: Personalizing quality criteria based on individual or organizational preferences
- **Performance-Based Calibration**: Aligning quality thresholds with achievable performance levels

**Advanced Critique Applications**

**Routing Optimization Feedback**
Using quality assessments to improve model selection strategies:
- **Model Suitability Analysis**: Identifying which models perform best for specific query types
- **Cost-Quality Trade-off Optimization**: Finding optimal balance points for different use cases
- **Fallback Strategy Refinement**: Improving fallback model selection based on quality maintenance
- **Resource Allocation Guidance**: Informing budget allocation based on quality return on investment

**System Performance Monitoring**
Comprehensive monitoring of overall system health and performance:
- **Quality Trend Analysis**: Tracking quality improvements or degradations over time
- **Performance Anomaly Detection**: Identifying unusual quality patterns requiring investigation
- **Comparative Performance Analysis**: Benchmarking against historical performance and industry standards
- **Proactive Quality Assurance**: Predicting and preventing quality issues before they impact users

**Quality Assurance Integration**

**Production Quality Gates**
Automated quality control mechanisms ensuring consistent output standards:
- **Minimum Quality Thresholds**: Automatic rejection or rerouting of responses below quality standards
- **Quality-Based Retry Logic**: Attempting response regeneration when quality assessments indicate problems
- **Progressive Quality Enhancement**: Iteratively improving responses through multiple critique cycles
- **Quality Certification**: Marking high-quality responses for reuse and reference

**User Experience Optimization**
Ensuring quality assessment contributes to improved user satisfaction:
- **Transparent Quality Indicators**: Providing users with quality confidence scores and assessments
- **Quality-Based Response Ranking**: Prioritizing higher-quality responses in multi-option scenarios
- **Personalized Quality Preferences**: Adapting quality standards to individual user preferences and needs
- **Quality Feedback Integration**: Incorporating user quality ratings into system improvement processes

This comprehensive quality assessment framework ensures that resource-aware optimization systems maintain high standards while continuously improving through intelligent feedback and adaptation mechanisms.`
    },
    {
      title: 'Advanced Optimization Techniques and Production Deployment',
      content: `Advanced optimization techniques for resource-aware systems encompass sophisticated strategies beyond basic model selection, including contextual pruning, proactive resource prediction, learned allocation policies, and enterprise-grade production deployment patterns.

**Contextual Pruning and Content Optimization**

**Intelligent Context Management**
Advanced techniques for minimizing token usage while preserving essential information:
- **Semantic Compression**: Removing redundant information while maintaining meaning and context
- **Relevance-Based Filtering**: Keeping only the most pertinent context for specific queries
- **Hierarchical Information Pruning**: Removing less critical details while preserving essential structure
- **Dynamic Context Window Management**: Adjusting context size based on query complexity and model capabilities

**Advanced Pruning Algorithms**
Sophisticated approaches to context optimization:
\`\`\`python
class ContextualPruningSystem:
    def __init__(self):
        self.importance_models = {
            'semantic_importance': SemanticImportanceModel(),
            'query_relevance': QueryRelevanceModel(),
            'context_dependency': ContextDependencyModel()
        }
        
        self.pruning_strategies = {
            'aggressive': {'compression_ratio': 0.3, 'quality_threshold': 0.7},
            'balanced': {'compression_ratio': 0.5, 'quality_threshold': 0.8},
            'conservative': {'compression_ratio': 0.7, 'quality_threshold': 0.9}
        }
        
    def optimize_context(self, context, query, target_tokens, strategy='balanced'):
        """Optimize context for token efficiency while preserving relevance."""
        
        # Analyze context components
        components = self._segment_context(context)
        
        # Score each component for importance
        importance_scores = {}
        for component_id, component_text in components.items():
            scores = {}
            
            # Semantic importance
            scores['semantic'] = self.importance_models['semantic_importance'].score(
                component_text, context
            )
            
            # Query relevance
            scores['relevance'] = self.importance_models['query_relevance'].score(
                component_text, query
            )
            
            # Context dependency
            scores['dependency'] = self.importance_models['context_dependency'].score(
                component_text, components
            )
            
            # Weighted combination
            importance_scores[component_id] = (
                scores['semantic'] * 0.3 +
                scores['relevance'] * 0.5 +
                scores['dependency'] * 0.2
            )
        
        # Select components based on strategy
        strategy_config = self.pruning_strategies[strategy]
        target_ratio = strategy_config['compression_ratio']
        
        # Sort by importance and select top components
        sorted_components = sorted(importance_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        selected_components = []
        current_tokens = 0
        
        for component_id, importance in sorted_components:
            component_tokens = self._estimate_tokens(components[component_id])
            
            if current_tokens + component_tokens <= target_tokens:
                selected_components.append(component_id)
                current_tokens += component_tokens
            elif current_tokens < target_tokens * target_ratio:
                # Try to include partial component if there's significant space
                partial_component = self._truncate_component(
                    components[component_id], target_tokens - current_tokens
                )
                selected_components.append(component_id)
                components[component_id] = partial_component
                break
        
        # Reconstruct optimized context
        optimized_context = self._reconstruct_context(components, selected_components)
        
        # Validate optimization quality
        quality_score = self._assess_pruning_quality(
            context, optimized_context, query
        )
        
        return {
            'optimized_context': optimized_context,
            'original_tokens': self._estimate_tokens(context),
            'optimized_tokens': self._estimate_tokens(optimized_context),
            'compression_ratio': current_tokens / self._estimate_tokens(context),
            'quality_score': quality_score,
            'components_retained': len(selected_components),
            'components_total': len(components)
        }
        
    def _assess_pruning_quality(self, original, pruned, query):
        """Assess quality loss from context pruning."""
        
        # Use lightweight model to compare context utility
        original_relevance = self._calculate_context_relevance(original, query)
        pruned_relevance = self._calculate_context_relevance(pruned, query)
        
        quality_retention = pruned_relevance / max(original_relevance, 0.1)
        return min(1.0, quality_retention)
\`\`\`

**Adaptive Content Summarization**
Dynamic summarization based on resource constraints and quality requirements:
- **Query-Focused Summarization**: Tailoring summaries to specific information needs
- **Progressive Detail Reduction**: Gradually reducing detail level based on token budget constraints
- **Key Information Preservation**: Ensuring critical facts and relationships are maintained
- **Quality-Aware Compression**: Balancing compression ratio with information preservation

**Proactive Resource Prediction and Planning**

**Workload Forecasting Systems**
Advanced prediction models for resource demand planning:
- **Time Series Analysis**: Predicting usage patterns based on historical data and seasonal trends
- **User Behavior Modeling**: Understanding individual and group usage patterns for capacity planning
- **Event-Driven Prediction**: Forecasting resource spikes based on scheduled events and business cycles
- **External Factor Integration**: Incorporating market conditions, holidays, and external events in predictions

**Intelligent Resource Pre-allocation**
Proactive resource management strategies:
- **Predictive Scaling**: Auto-scaling resources based on forecasted demand rather than reactive scaling
- **Resource Pooling**: Maintaining shared resource pools for efficient allocation across different use cases
- **Peak Load Preparation**: Pre-provisioning resources for predicted high-demand periods
- **Capacity Reservation**: Strategic reservation of computational resources during high-availability periods

**Learned Resource Allocation Policies**

**Reinforcement Learning for Optimization**
Machine learning approaches to resource allocation policy development:
- **Multi-Armed Bandit Algorithms**: Learning optimal model selection through exploration and exploitation
- **Policy Gradient Methods**: Optimizing allocation strategies through continuous policy improvement
- **Q-Learning Adaptation**: Learning state-action values for different resource allocation scenarios
- **Actor-Critic Systems**: Combining value estimation with policy optimization for robust learning

**Adaptive Optimization Algorithms**
Self-improving systems that learn from operational experience:
- **Performance Feedback Integration**: Incorporating quality assessments and user satisfaction into learning
- **Cost-Benefit Learning**: Understanding true cost-effectiveness of different optimization strategies
- **Context-Sensitive Adaptation**: Learning different optimization approaches for different types of queries
- **Multi-Objective Optimization**: Balancing multiple objectives (cost, quality, speed) through learned policies

**Production Deployment Architecture**

**Enterprise-Scale Infrastructure**
Production-ready architectures for large-scale resource-aware systems:
- **Microservices Architecture**: Decomposing optimization components into scalable, maintainable services
- **Container Orchestration**: Using Kubernetes and similar systems for dynamic scaling and management
- **Service Mesh Integration**: Implementing service communication, monitoring, and security at scale
- **Multi-Region Deployment**: Geographic distribution for performance, compliance, and disaster recovery

**High-Availability Design Patterns**
Ensuring system reliability and continuous operation:
- **Circuit Breaker Patterns**: Preventing cascading failures through intelligent failure detection
- **Redundancy and Failover**: Multiple backup systems and automatic failover mechanisms
- **Graceful Degradation**: Maintaining basic functionality during partial system failures
- **Health Monitoring**: Comprehensive health checks and automatic recovery procedures

**Performance Monitoring and Optimization**

**Comprehensive Observability**
Production monitoring systems for resource-aware optimization:
- **Real-Time Metrics**: Live dashboards showing resource utilization, costs, and performance
- **Distributed Tracing**: End-to-end request tracing across complex optimization pipelines
- **Custom Business Metrics**: Tracking optimization-specific KPIs and business outcomes
- **Anomaly Detection**: Automated detection of unusual patterns requiring investigation

**Continuous Improvement Processes**
Ongoing optimization of production systems:
- **Performance Benchmarking**: Regular assessment against performance baselines and targets
- **A/B Testing Frameworks**: Controlled testing of optimization strategies and improvements
- **Capacity Planning**: Regular assessment and planning for system growth and scaling needs
- **Cost Optimization Reviews**: Periodic analysis and optimization of resource costs and efficiency

**Security and Compliance Considerations**

**Resource Access Security**
Protecting access to computational resources and models:
- **API Security**: Comprehensive authentication, authorization, and rate limiting
- **Resource Isolation**: Ensuring tenant isolation and preventing resource access violations
- **Audit Logging**: Complete audit trails for resource access and optimization decisions
- **Compliance Monitoring**: Ensuring optimization decisions meet regulatory and policy requirements

**Data Privacy in Optimization**
Protecting sensitive information during optimization processes:
- **Query Anonymization**: Removing or masking personally identifiable information
- **Federated Optimization**: Enabling optimization without centralizing sensitive data
- **Privacy-Preserving Analytics**: Using differential privacy and other techniques for safe analytics
- **Regulatory Compliance**: Ensuring GDPR, CCPA, and other privacy regulation compliance

This comprehensive framework for advanced optimization techniques ensures that resource-aware systems can achieve maximum efficiency while maintaining security, compliance, and reliability at enterprise scale.`
    }
  ],

  practicalExamples: [
    {
      title: 'Financial Services Multi-Tier Resource Optimization',
      description: 'Large financial institution implementing resource-aware optimization across trading, risk analysis, and customer service with strict cost controls and quality requirements',
      example: 'Investment bank deploying tiered model selection for different financial analysis tasks with real-time cost monitoring and quality assurance',
      steps: [
        'Router Agent Architecture: Implement sophisticated query classification system analyzing financial complexity levels from simple account inquiries to complex derivative pricing models',
        'Cost-Quality Optimization: Deploy dynamic model selection balancing expensive high-precision models for trading decisions against fast affordable models for routine customer queries',
        'Budget Management Framework: Establish multi-level budget controls with department allocations, real-time spend monitoring, and automatic cost protection mechanisms',
        'Quality Assurance Integration: Implement critique agents evaluating financial analysis accuracy with domain-specific metrics and regulatory compliance validation',
        'Fallback and Recovery Systems: Design comprehensive fallback mechanisms ensuring trading systems remain operational with degraded but acceptable performance during model outages',
        'Performance Monitoring: Deploy real-time analytics tracking cost efficiency, response quality, regulatory compliance, and risk management effectiveness across all financial operations'
      ]
    },
    {
      title: 'Healthcare Resource-Aware Clinical Decision Support',
      description: 'Hospital network implementing intelligent resource optimization for clinical decision support while maintaining patient safety and regulatory compliance',
      steps: [
        'Clinical Query Classification: Develop medical complexity analysis routing routine administrative queries to fast models while directing critical diagnostic questions to premium models',
        'Safety-First Resource Management: Implement quality thresholds ensuring life-critical medical decisions always receive maximum computational resources regardless of cost constraints',
        'HIPAA-Compliant Optimization: Deploy privacy-preserving resource optimization ensuring patient data protection while enabling cost-effective clinical decision support',
        'Multi-Modal Medical Integration: Optimize resource allocation across text analysis, medical imaging processing, and structured clinical data analysis with appropriate model selection',
        'Emergency Response Protocols: Establish high-priority resource allocation for emergency medical situations bypassing normal cost controls and routing constraints',
        'Clinical Quality Validation: Implement medical expert validation systems ensuring AI-generated clinical recommendations meet healthcare quality standards and regulatory requirements'
      ]
    },
    {
      title: 'E-commerce Platform Dynamic Resource Scaling',
      description: 'Global e-commerce company implementing resource-aware optimization for customer support, product recommendations, and content generation with seasonal demand variations',
      example: 'Online retailer managing Black Friday traffic spikes while optimizing costs during low-demand periods through intelligent resource allocation',
      steps: [
        'Seasonal Demand Forecasting: Implement predictive models forecasting resource needs based on shopping patterns, seasonal trends, and promotional events',
        'Customer Tier-Based Optimization: Deploy tiered service levels routing VIP customers to premium models while using cost-effective models for standard customer interactions',
        'Real-Time Load Balancing: Create dynamic resource allocation adjusting model selection based on current system load, API availability, and response time requirements',
        'Content Generation Optimization: Optimize product description generation, review summarization, and personalized recommendations through intelligent model selection and contextual pruning',
        'Geographic Resource Distribution: Implement multi-region resource optimization considering local costs, latency requirements, and data residency regulations',
        'Revenue-Driven Priority Management: Establish resource allocation policies prioritizing high-value transactions and premium customer interactions during peak demand periods'
      ]
    }
  ],

  references: [
    'Google Agent Development Kit (ADK) Documentation. https://google.github.io/adk-docs/',
    'Gemini Flash 2.0 & Gemini Pro 2.0 Model Documentation. https://aistudio.google.com/',
    'OpenRouter Unified AI Model Access. https://openrouter.ai/docs/quickstart',
    'Syed, M. (2025). Resource-Aware Optimization LLM Implementation. GitHub. https://github.com/mahtabsyed/21-Agentic-Patterns/',
    'OpenAI API Documentation and Pricing. https://platform.openai.com/docs/api-reference',
    'Google Custom Search API Documentation. https://developers.google.com/custom-search/v1/introduction',
    'Cost Optimization in Large Language Model Applications. arXiv preprint arXiv:2401.15742'
  ],

  navigation: {
    previous: { href: '/chapters/inter-agent-communication', title: 'Inter-Agent Communication (A2A)' },
    next: { href: '/chapters/reasoning-techniques', title: 'Reasoning Techniques' }
  }
}
