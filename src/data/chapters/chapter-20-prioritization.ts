import { Chapter } from '../types';

export const prioritizationChapter: Chapter = {
  id: 'prioritization',
  title: 'Prioritization',
  subtitle: 'Intelligent Task Management and Decision-Making Under Resource Constraints',
  description: 'Implement sophisticated prioritization frameworks enabling agents to assess, rank, and dynamically manage multiple competing tasks and objectives for optimal resource allocation.',
  readingTime: '29 min read',
  overview: `In complex, dynamic environments, agents frequently encounter numerous potential actions, conflicting goals, and limited resources. Without a defined process for determining the subsequent action, agents may experience reduced efficiency, operational delays, or failures to achieve key objectives. The prioritization pattern addresses this issue by enabling agents to assess and rank tasks, objectives, or actions based on their significance, urgency, dependencies, and established criteria.

This ensures agents concentrate efforts on the most critical tasks, resulting in enhanced effectiveness and goal alignment. Prioritization facilitates informed decision-making when addressing multiple demands, prioritizing vital or urgent activities over less critical ones. It is particularly relevant in real-world scenarios where resources are constrained, time is limited, and objectives may conflict.

The fundamental aspects of agent prioritization involve criteria definition, task evaluation, scheduling logic, and dynamic re-prioritization. This capability enables agents to exhibit more intelligent, efficient, and robust behavior, especially in complex, multi-objective environments, mirroring human team organization where managers prioritize tasks by considering input from all members.`,
  keyPoints: [
    'Multi-criteria decision-making frameworks incorporating urgency, importance, dependencies, resource availability, and cost-benefit analysis for comprehensive task evaluation',
    'Dynamic re-prioritization capabilities enabling real-time adaptation to changing circumstances, emerging critical events, and shifting deadline constraints',
    'Hierarchical prioritization at multiple levels from strategic goal selection to tactical sub-task ordering and immediate action selection for comprehensive decision-making',
    'Resource-aware scheduling algorithms optimizing task allocation based on computational resources, time constraints, and worker availability in distributed systems',
    'LangChain-based project management agents demonstrating practical implementation of intelligent task creation, priority assignment, and team coordination',
    'Context-aware priority adjustment mechanisms responding to user preferences, system state changes, and external environmental factors',
    'Integration with existing planning and execution systems enabling seamless workflow management and strategic alignment across complex agent architectures',
    'Performance optimization through intelligent queue management, dependency resolution, and parallel task execution strategies for maximum operational efficiency'
  ],
  codeExample: `# Comprehensive Prioritization Framework for Intelligent Agents
# Advanced task management and decision-making system with dynamic prioritization

import os
import asyncio
import json
import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import math

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriorityLevel(Enum):
    """Priority levels for task classification."""
    CRITICAL = "P0"    # Immediate attention required
    HIGH = "P1"        # High importance, time-sensitive
    MEDIUM = "P2"      # Normal priority
    LOW = "P3"         # Can be deferred

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class ResourceType(Enum):
    """Types of resources required for task execution."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    HUMAN = "human"
    EXTERNAL_API = "external_api"

# --- Advanced Task Management Models ---
@dataclass
class TaskDependency:
    """Represents a dependency relationship between tasks."""
    prerequisite_id: str
    dependency_type: str = "blocks"  # blocks, enables, optimizes
    strength: float = 1.0  # 0.0 to 1.0, importance of dependency

@dataclass
class ResourceRequirement:
    """Resource requirements for task execution."""
    resource_type: ResourceType
    amount: float
    duration_minutes: float
    is_exclusive: bool = False

class AdvancedTask(BaseModel):
    """Comprehensive task representation with prioritization metadata."""
    id: str
    title: str
    description: str
    priority_level: PriorityLevel = PriorityLevel.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Prioritization factors
    urgency_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Time sensitivity (0.0-1.0)")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Strategic value (0.0-1.0)")
    effort_estimate: float = Field(default=1.0, gt=0.0, description="Estimated hours to complete")
    
    # Scheduling information
    created_at: datetime = Field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    assigned_to: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    
    # Dependencies and resources
    dependencies: List[TaskDependency] = Field(default_factory=list)
    resource_requirements: List[ResourceRequirement] = Field(default_factory=list)
    
    # Contextual metadata
    tags: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    def calculate_priority_score(self) -> float:
        """Calculate composite priority score based on multiple factors."""
        base_score = (self.urgency_score * 0.4 + self.importance_score * 0.6)
        
        # Deadline urgency multiplier
        deadline_multiplier = 1.0
        if self.deadline:
            time_to_deadline = (self.deadline - datetime.now()).total_seconds() / 3600  # hours
            if time_to_deadline < 24:
                deadline_multiplier = 2.0
            elif time_to_deadline < 168:  # 1 week
                deadline_multiplier = 1.5
        
        # Effort efficiency (higher priority for quick wins)
        effort_factor = 1.0 / math.sqrt(max(self.effort_estimate, 0.5))
        
        return base_score * deadline_multiplier * effort_factor

class ComprehensivePrioritizationEngine:
    """
    Advanced prioritization system supporting multi-criteria decision making,
    dynamic re-prioritization, and resource-aware scheduling.
    """
    
    def __init__(self):
        self.tasks: Dict[str, AdvancedTask] = {}
        self.task_queue: List[Tuple[float, str]] = []  # (priority_score, task_id)
        self.resource_pool: Dict[ResourceType, float] = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.HUMAN: 8.0,  # 8 hours per day
            ResourceType.EXTERNAL_API: 1000.0  # API calls per hour
        }
        self.active_tasks: Dict[str, datetime] = {}
        self.priority_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("PrioritizationEngine")
        self.logger.info("Advanced Prioritization Engine initialized")
    
    def create_task(self, 
                   title: str,
                   description: str,
                   urgency: float = 0.5,
                   importance: float = 0.5,
                   effort_hours: float = 1.0,
                   deadline: Optional[datetime] = None,
                   assigned_to: Optional[str] = None,
                   tags: List[str] = None,
                   dependencies: List[str] = None) -> AdvancedTask:
        """Create a new task with comprehensive prioritization metadata."""
        
        task_id = f"TASK-{len(self.tasks) + 1:04d}"
        
        # Process dependencies
        task_dependencies = []
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.tasks:
                    task_dependencies.append(TaskDependency(prerequisite_id=dep_id))
        
        task = AdvancedTask(
            id=task_id,
            title=title,
            description=description,
            urgency_score=urgency,
            importance_score=importance,
            effort_estimate=effort_hours,
            deadline=deadline,
            assigned_to=assigned_to,
            tags=tags or [],
            dependencies=task_dependencies
        )
        
        self.tasks[task_id] = task
        self._update_priority_queue()
        
        self.logger.info(f"Created task {task_id}: {title} (Priority Score: {task.calculate_priority_score():.3f})")
        return task
    
    def _update_priority_queue(self):
        """Rebuild priority queue based on current task priorities."""
        self.task_queue.clear()
        
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(task):
                priority_score = -task.calculate_priority_score()  # Negative for max heap behavior
                heapq.heappush(self.task_queue, (priority_score, task_id))
        
        self.logger.debug(f"Updated priority queue with {len(self.task_queue)} ready tasks")
    
    def _are_dependencies_satisfied(self, task: AdvancedTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dependency in task.dependencies:
            prerequisite = self.tasks.get(dependency.prerequisite_id)
            if not prerequisite or prerequisite.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def get_next_task(self, worker_id: Optional[str] = None) -> Optional[AdvancedTask]:
        """Get the highest priority task ready for execution."""
        self._update_priority_queue()
        
        while self.task_queue:
            priority_score, task_id = heapq.heappop(self.task_queue)
            task = self.tasks.get(task_id)
            
            if not task or task.status != TaskStatus.PENDING:
                continue
            
            # Check worker assignment
            if worker_id and task.assigned_to and task.assigned_to != worker_id:
                continue
            
            # Check resource availability
            if self._check_resource_availability(task):
                self.logger.info(f"Selected next task: {task.id} ({task.title}) for worker {worker_id or 'any'}")
                return task
        
        return None
    
    def _check_resource_availability(self, task: AdvancedTask) -> bool:
        """Check if required resources are available for task execution."""
        for requirement in task.resource_requirements:
            available = self.resource_pool.get(requirement.resource_type, 0.0)
            if available < requirement.amount:
                return False
        return True
    
    def start_task(self, task_id: str) -> bool:
        """Mark task as in progress and allocate resources."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        # Allocate resources
        for requirement in task.resource_requirements:
            self.resource_pool[requirement.resource_type] -= requirement.amount
        
        task.status = TaskStatus.IN_PROGRESS
        self.active_tasks[task_id] = datetime.now()
        task.estimated_completion = datetime.now() + timedelta(hours=task.effort_estimate)
        
        self.logger.info(f"Started task {task_id}: {task.title}")
        return True
    
    def complete_task(self, task_id: str) -> bool:
        """Mark task as completed and free resources."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.IN_PROGRESS:
            return False
        
        # Free resources
        for requirement in task.resource_requirements:
            self.resource_pool[requirement.resource_type] += requirement.amount
        
        task.status = TaskStatus.COMPLETED
        self.active_tasks.pop(task_id, None)
        self._update_priority_queue()  # May unblock dependent tasks
        
        self.logger.info(f"Completed task {task_id}: {task.title}")
        return True
    
    def dynamic_reprioritize(self, event_type: str, context: Dict[str, Any]):
        """Dynamically adjust task priorities based on changing conditions."""
        
        reprioritization_rules = {
            "deadline_approaching": self._handle_deadline_urgency,
            "resource_shortage": self._handle_resource_constraints,
            "new_critical_task": self._handle_critical_insertion,
            "user_preference_change": self._handle_preference_update,
            "external_dependency": self._handle_external_changes
        }
        
        handler = reprioritization_rules.get(event_type)
        if handler:
            changes = handler(context)
            if changes:
                self._update_priority_queue()
                self._log_reprioritization(event_type, changes)
    
    def _handle_deadline_urgency(self, context: Dict[str, Any]) -> List[str]:
        """Handle approaching deadlines by boosting urgency scores."""
        changes = []
        current_time = datetime.now()
        
        for task_id, task in self.tasks.items():
            if task.deadline and task.status == TaskStatus.PENDING:
                time_remaining = (task.deadline - current_time).total_seconds() / 3600
                
                if time_remaining < 24 and task.urgency_score < 0.9:
                    old_urgency = task.urgency_score
                    task.urgency_score = min(0.95, task.urgency_score + 0.3)
                    changes.append(f"{task_id}: urgency {old_urgency:.2f} -> {task.urgency_score:.2f}")
        
        return changes
    
    def _handle_resource_constraints(self, context: Dict[str, Any]) -> List[str]:
        """Handle resource shortages by prioritizing efficient tasks."""
        changes = []
        constrained_resource = context.get("resource_type")
        
        if constrained_resource:
            # Boost priority for tasks that use less of the constrained resource
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.PENDING:
                    for req in task.resource_requirements:
                        if req.resource_type.value == constrained_resource and req.amount < 5.0:
                            old_importance = task.importance_score
                            task.importance_score = min(1.0, task.importance_score + 0.2)
                            changes.append(f"{task_id}: importance {old_importance:.2f} -> {task.importance_score:.2f}")
        
        return changes
    
    def _handle_critical_insertion(self, context: Dict[str, Any]) -> List[str]:
        """Handle insertion of new critical tasks."""
        return ["Critical task handling not implemented in demo"]
    
    def _handle_preference_update(self, context: Dict[str, Any]) -> List[str]:
        """Handle user preference changes."""
        return ["Preference update handling not implemented in demo"]
    
    def _handle_external_changes(self, context: Dict[str, Any]) -> List[str]:
        """Handle external dependency changes."""
        return ["External dependency handling not implemented in demo"]
    
    def _log_reprioritization(self, event_type: str, changes: List[str]):
        """Log reprioritization events for analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "changes": changes,
            "queue_size": len(self.task_queue)
        }
        self.priority_history.append(log_entry)
        self.logger.info(f"Reprioritization ({event_type}): {len(changes)} tasks adjusted")
    
    def get_prioritization_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive prioritization analytics."""
        total_tasks = len(self.tasks)
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
            priority_counts[task.priority_level.value] += 1
        
        avg_queue_time = 0.0
        if self.active_tasks:
            current_time = datetime.now()
            total_queue_time = sum(
                (current_time - start_time).total_seconds() / 60
                for start_time in self.active_tasks.values()
            )
            avg_queue_time = total_queue_time / len(self.active_tasks)
        
        return {
            "total_tasks": total_tasks,
            "status_distribution": dict(status_counts),
            "priority_distribution": dict(priority_counts),
            "ready_tasks_in_queue": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "average_queue_time_minutes": avg_queue_time,
            "resource_utilization": {
                resource.value: 100.0 - available 
                for resource, available in self.resource_pool.items()
            },
            "reprioritization_events": len(self.priority_history)
        }
    
    def get_task_recommendations(self, worker_id: str, max_recommendations: int = 5) -> List[AdvancedTask]:
        """Get personalized task recommendations for a specific worker."""
        candidate_tasks = []
        
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                self._are_dependencies_satisfied(task) and
                (not task.assigned_to or task.assigned_to == worker_id)):
                candidate_tasks.append(task)
        
        # Sort by priority score
        candidate_tasks.sort(key=lambda t: t.calculate_priority_score(), reverse=True)
        
        return candidate_tasks[:max_recommendations]

# --- LangChain Project Manager Integration ---
class ProjectManagerArgs(BaseModel):
    """Base arguments for project manager tools."""
    pass

class CreateTaskArgs(ProjectManagerArgs):
    description: str = Field(description="Detailed task description")
    urgency: Optional[float] = Field(default=0.5, description="Urgency score 0.0-1.0")
    importance: Optional[float] = Field(default=0.5, description="Importance score 0.0-1.0")
    effort_hours: Optional[float] = Field(default=1.0, description="Estimated effort in hours")

class UpdatePriorityArgs(ProjectManagerArgs):
    task_id: str = Field(description="Task ID to update")
    urgency: Optional[float] = Field(description="New urgency score 0.0-1.0")
    importance: Optional[float] = Field(description="New importance score 0.0-1.0")

class AssignTaskArgs(ProjectManagerArgs):
    task_id: str = Field(description="Task ID to assign")
    worker_id: str = Field(description="Worker identifier")

class IntelligentProjectManager:
    """
    LangChain-based project manager with advanced prioritization capabilities.
    """
    
    def __init__(self, openai_api_key: str = None):
        self.prioritization_engine = ComprehensivePrioritizationEngine()
        self.workers = ["Alice", "Bob", "Charlie", "ReviewTeam"]
        
        # Initialize LLM
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=api_key)
        
        # Setup tools and agent
        self._setup_tools()
        self._setup_agent()
        
        self.logger = logging.getLogger("IntelligentProjectManager")
        self.logger.info("Intelligent Project Manager initialized")
    
    def _setup_tools(self):
        """Setup project management tools for the LangChain agent."""
        
        def create_intelligent_task(description: str, urgency: float = 0.5, 
                                  importance: float = 0.5, effort_hours: float = 1.0) -> str:
            task = self.prioritization_engine.create_task(
                title=description[:50] + "..." if len(description) > 50 else description,
                description=description,
                urgency=urgency,
                importance=importance,
                effort_hours=effort_hours
            )
            return f"Created task {task.id} with priority score {task.calculate_priority_score():.3f}"
        
        def update_task_priority(task_id: str, urgency: float = None, importance: float = None) -> str:
            task = self.prioritization_engine.tasks.get(task_id)
            if not task:
                return f"Task {task_id} not found"
            
            old_score = task.calculate_priority_score()
            if urgency is not None:
                task.urgency_score = max(0.0, min(1.0, urgency))
            if importance is not None:
                task.importance_score = max(0.0, min(1.0, importance))
            
            new_score = task.calculate_priority_score()
            self.prioritization_engine._update_priority_queue()
            
            return f"Updated {task_id} priority: {old_score:.3f} -> {new_score:.3f}"
        
        def assign_task_to_worker(task_id: str, worker_id: str) -> str:
            task = self.prioritization_engine.tasks.get(task_id)
            if not task:
                return f"Task {task_id} not found"
            
            if worker_id in self.workers:
                task.assigned_to = worker_id
                return f"Assigned {task_id} to {worker_id}"
            else:
                return f"Unknown worker {worker_id}. Available: {', '.join(self.workers)}"
        
        def get_next_priority_task(worker_id: str = None) -> str:
            task = self.prioritization_engine.get_next_task(worker_id)
            if task:
                return f"Next task for {worker_id or 'anyone'}: {task.id} - {task.title} (Score: {task.calculate_priority_score():.3f})"
            else:
                return f"No tasks available for {worker_id or 'anyone'}"
        
        def list_all_tasks_with_priorities() -> str:
            if not self.prioritization_engine.tasks:
                return "No tasks in the system"
            
            tasks_info = []
            for task in sorted(self.prioritization_engine.tasks.values(), 
                             key=lambda t: t.calculate_priority_score(), reverse=True):
                score = task.calculate_priority_score()
                tasks_info.append(
                    f"{task.id}: {task.title} | Priority: {score:.3f} | "
                    f"Status: {task.status.value} | Assigned: {task.assigned_to or 'None'}"
                )
            
            return "Tasks (by priority):\\n" + "\\n".join(tasks_info)
        
        def trigger_dynamic_reprioritization(event_type: str, context_json: str = "{}") -> str:
            try:
                context = json.loads(context_json)
                self.prioritization_engine.dynamic_reprioritize(event_type, context)
                return f"Triggered reprioritization for event: {event_type}"
            except json.JSONDecodeError:
                return "Invalid context JSON provided"
        
        self.tools = [
            Tool(
                name="create_intelligent_task",
                func=create_intelligent_task,
                description="Create a new task with intelligent priority scoring",
                args_schema=CreateTaskArgs
            ),
            Tool(
                name="update_task_priority",
                func=update_task_priority,
                description="Update task urgency and importance scores",
                args_schema=UpdatePriorityArgs
            ),
            Tool(
                name="assign_task_to_worker",
                func=assign_task_to_worker,
                description="Assign a task to a specific worker",
                args_schema=AssignTaskArgs
            ),
            Tool(
                name="get_next_priority_task",
                func=get_next_priority_task,
                description="Get the next highest priority task for a worker"
            ),
            Tool(
                name="list_all_tasks_with_priorities",
                func=list_all_tasks_with_priorities,
                description="List all tasks ordered by priority score"
            ),
            Tool(
                name="trigger_dynamic_reprioritization", 
                func=trigger_dynamic_reprioritization,
                description="Trigger dynamic reprioritization based on events"
            )
        ]
    
    def _setup_agent(self):
        """Setup the LangChain agent with prioritization capabilities."""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an Intelligent Project Manager specializing in advanced task prioritization and resource management.

Your capabilities include:
- Creating tasks with intelligent priority scoring based on urgency, importance, and effort
- Dynamically adjusting priorities based on changing conditions
- Optimal task assignment considering worker skills and availability
- Real-time reprioritization for critical events

Available workers: Alice, Bob, Charlie, ReviewTeam

When creating tasks:
1. Analyze the request for urgency indicators (ASAP, urgent, critical, deadline)
2. Assess importance based on business impact or strategic value
3. Estimate effort realistically
4. Assign to appropriate workers based on task type

Priority Levels:
- Urgency: 0.9+ (Critical deadlines), 0.7+ (Time-sensitive), 0.5 (Normal), 0.3- (Flexible)
- Importance: 0.9+ (Strategic/Revenue), 0.7+ (Important features), 0.5 (Normal), 0.3- (Nice-to-have)

Always show the final prioritized task list after making changes."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_react_agent(self.llm, self.tools, prompt_template)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            max_iterations=10
        )
    
    async def process_request(self, request: str) -> str:
        """Process a project management request through the intelligent agent."""
        try:
            result = await self.agent_executor.ainvoke({"input": request})
            return result.get("output", "No response generated")
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return f"Error: {str(e)}"
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive prioritization analytics report."""
        return self.prioritization_engine.get_prioritization_analytics()

# --- Demonstration Functions ---
async def demonstrate_intelligent_prioritization():
    """Demonstrate comprehensive prioritization capabilities."""
    
    print("🎯 Intelligent Prioritization System Demonstration")
    print("=" * 65)
    
    # Initialize project manager (would need OPENAI_API_KEY in real usage)
    print("\\n🔧 Initializing Intelligent Project Manager...")
    try:
        pm = IntelligentProjectManager()
        print("✅ Project Manager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Project Manager: {e}")
        print("💡 Demo will continue with engine-only examples...")
        pm = None
    
    # Direct engine demonstration
    print("\\n🚀 Direct Prioritization Engine Demo")
    print("-" * 45)
    
    engine = ComprehensivePrioritizationEngine()
    
    # Create various tasks with different characteristics
    tasks_to_create = [
        {
            "title": "Fix critical security vulnerability",
            "description": "Patch SQL injection vulnerability in user authentication",
            "urgency": 0.95,
            "importance": 0.9,
            "effort_hours": 2.0,
            "deadline": datetime.now() + timedelta(hours=4),
            "assigned_to": "Alice"
        },
        {
            "title": "Implement user dashboard",
            "description": "Create responsive dashboard for user analytics",
            "urgency": 0.6,
            "importance": 0.8,
            "effort_hours": 8.0,
            "deadline": datetime.now() + timedelta(days=7),
            "assigned_to": "Bob"
        },
        {
            "title": "Update documentation",
            "description": "Refresh API documentation and examples",
            "urgency": 0.3,
            "importance": 0.4,
            "effort_hours": 3.0,
            "assigned_to": "Charlie"
        },
        {
            "title": "Performance optimization",
            "description": "Optimize database queries for better response times",
            "urgency": 0.7,
            "importance": 0.7,
            "effort_hours": 5.0,
            "deadline": datetime.now() + timedelta(days=3)
        }
    ]
    
    print("Creating tasks with different priority characteristics:")
    created_tasks = []
    for task_spec in tasks_to_create:
        task = engine.create_task(**task_spec)
        created_tasks.append(task)
        print(f"  📋 {task.id}: {task.title}")
        print(f"      Priority Score: {task.calculate_priority_score():.3f}")
        print(f"      Urgency: {task.urgency_score:.2f}, Importance: {task.importance_score:.2f}")
    
    print("\\n📊 Task Prioritization Analysis")
    print("-" * 35)
    
    # Show priority queue
    print("\\nPriority Queue (Ready Tasks):")
    while True:
        next_task = engine.get_next_task()
        if not next_task:
            break
        print(f"  🥇 {next_task.id}: {next_task.title}")
        print(f"      Score: {next_task.calculate_priority_score():.3f}")
        print(f"      Assigned to: {next_task.assigned_to or 'Unassigned'}")
        
        # Simulate starting the task
        engine.start_task(next_task.id)
        break  # Just show the top task
    
    print("\\n⚡ Dynamic Reprioritization Demo")
    print("-" * 35)
    
    # Simulate deadline approaching
    print("\\n📅 Simulating approaching deadlines...")
    engine.dynamic_reprioritize("deadline_approaching", {})
    
    # Show updated priorities
    print("\\nUpdated Task Priorities:")
    for task in sorted(created_tasks, key=lambda t: t.calculate_priority_score(), reverse=True):
        print(f"  {task.id}: {task.title}")
        print(f"    Score: {task.calculate_priority_score():.3f} (U:{task.urgency_score:.2f}, I:{task.importance_score:.2f})")
    
    # Analytics report
    print("\\n📈 System Analytics")
    print("-" * 20)
    analytics = engine.get_prioritization_analytics()
    
    print(f"Total Tasks: {analytics['total_tasks']}")
    print(f"Ready Tasks in Queue: {analytics['ready_tasks_in_queue']}")
    print(f"Active Tasks: {analytics['active_tasks']}")
    print(f"Reprioritization Events: {analytics['reprioritization_events']}")
    
    print("\\nStatus Distribution:")
    for status, count in analytics['status_distribution'].items():
        print(f"  {status}: {count}")
    
    print("\\nResource Utilization:")
    for resource, utilization in analytics['resource_utilization'].items():
        print(f"  {resource}: {utilization:.1f}%")
    
    # LangChain Agent Demo (if available)
    if pm:
        print("\\n🤖 LangChain Agent Integration Demo")
        print("-" * 40)
        
        demo_requests = [
            "Create an urgent task to fix the payment gateway bug. It's blocking customers and needs immediate attention from Alice.",
            "We need a new feature for user notifications. It's important for Q1 goals but not super urgent. Assign to Bob.",
            "Can you show me the current task priorities and suggest what Alice should work on next?"
        ]
        
        for i, request in enumerate(demo_requests, 1):
            print(f"\\n🗣️  User Request {i}:")
            print(f"'{request}'")
            print("\\n🤖 Agent Response:")
            
            try:
                response = await pm.process_request(request)
                print(response)
            except Exception as e:
                print(f"❌ Agent error: {e}")
        
        # Final analytics
        print("\\n📊 Final Analytics Report")
        print("-" * 30)
        final_analytics = pm.get_analytics_report()
        print(json.dumps(final_analytics, indent=2, default=str))
    
    print("\\n✅ Intelligent Prioritization Demonstration Complete!")
    print("Advanced task management enables optimal resource allocation and goal achievement")

# Example usage
if __name__ == "__main__":
    asyncio.run(demonstrate_intelligent_prioritization())`,
  sections: [
    {
      title: 'Multi-Criteria Decision-Making Framework',
      content: `Intelligent prioritization requires sophisticated evaluation frameworks that consider multiple competing factors simultaneously to make optimal resource allocation decisions.

**Core Prioritization Criteria**

Effective agent prioritization relies on systematic evaluation across key dimensions:

**Urgency Assessment**: Time sensitivity evaluation based on deadlines, external dependencies, and cascading impact potential
- Critical deadlines requiring immediate attention (urgency score: 0.9+)
- Time-sensitive tasks with approaching deadlines (0.7-0.9)
- Normal timeline tasks with flexible scheduling (0.3-0.7)
- Deferrable tasks with no immediate time pressure (0.0-0.3)

**Importance Evaluation**: Strategic value assessment considering business impact, user benefit, and long-term consequences
- Strategic initiatives with significant revenue or competitive impact
- Core functionality affecting primary user workflows
- Enhancement features improving user experience
- Maintenance tasks supporting system stability

**Resource Efficiency Analysis**: Cost-benefit evaluation considering computational resources, time investment, and expected outcomes
- Quick wins with high impact and low effort (prioritized for momentum)
- Resource-intensive tasks with proportional strategic value
- Maintenance activities with long-term operational benefits
- Research or exploration tasks with uncertain but potentially high returns

**Dependency Management**: Understanding task interdependencies and prerequisite relationships
- Blocking dependencies that prevent other tasks from starting
- Enabling dependencies that unlock multiple downstream activities
- Optional dependencies that optimize but don't require coordination
- Circular dependencies requiring careful scheduling and resolution

**Implementation Example:**

\`\`\`python
def calculate_priority_score(self) -> float:
    \"\"\"Calculate composite priority score based on multiple factors.\"\"\"
    base_score = (self.urgency_score * 0.4 + self.importance_score * 0.6)
    
    # Deadline urgency multiplier
    deadline_multiplier = 1.0
    if self.deadline:
        time_to_deadline = (self.deadline - datetime.now()).total_seconds() / 3600
        if time_to_deadline < 24:
            deadline_multiplier = 2.0
        elif time_to_deadline < 168:  # 1 week
            deadline_multiplier = 1.5
    
    # Effort efficiency (higher priority for quick wins)
    effort_factor = 1.0 / math.sqrt(max(self.effort_estimate, 0.5))
    
    return base_score * deadline_multiplier * effort_factor
\`\`\`

This multi-dimensional approach ensures balanced decision-making that considers both immediate operational needs and strategic long-term objectives.`
    },
    {
      title: 'Dynamic Re-Prioritization Mechanisms',
      content: `Dynamic re-prioritization enables agents to adapt their focus in real-time as conditions change, ensuring optimal resource allocation under evolving circumstances.

**Event-Driven Priority Adjustment**

Sophisticated agents monitor environmental changes and automatically adjust task priorities based on predefined rules and learned patterns:

**Deadline Proximity Response**: Automatic urgency escalation as deadlines approach
- 24-hour deadline: Boost urgency score by 0.3-0.5 points
- 1-week deadline: Apply 1.5x priority multiplier
- Overdue tasks: Maximum urgency with immediate scheduling

**Resource Constraint Adaptation**: Priority adjustment based on resource availability
- CPU/Memory constraints: Prioritize lightweight tasks for continued progress
- Network limitations: Defer external API-dependent tasks
- Human resource scarcity: Focus on automated or self-service capabilities

**External Event Integration**: Real-time adjustment based on external triggers
- Security incidents: Immediate escalation of security-related tasks
- Customer complaints: Dynamic promotion of user-facing fixes
- Market opportunities: Strategic task reprioritization for competitive advantage

**Cascade Impact Analysis**: Understanding how priority changes affect dependent tasks
- Upstream task completion: Automatic activation of dependent tasks
- Resource reallocation: Cascading schedule adjustments across task network
- Skill availability: Dynamic reassignment based on team member availability

**Learning-Based Adaptation**: Historical analysis informing future prioritization decisions
- Pattern recognition from past reprioritization events
- Success rate analysis of different priority strategies
- User feedback integration for preference learning

**Implementation Strategies:**

\`\`\`python
def dynamic_reprioritize(self, event_type: str, context: Dict[str, Any]):
    \"\"\"Dynamically adjust task priorities based on changing conditions.\"\"\"
    
    reprioritization_rules = {
        "deadline_approaching": self._handle_deadline_urgency,
        "resource_shortage": self._handle_resource_constraints,  
        "new_critical_task": self._handle_critical_insertion,
        "user_preference_change": self._handle_preference_update
    }
    
    handler = reprioritization_rules.get(event_type)
    if handler:
        changes = handler(context)
        if changes:
            self._update_priority_queue()
            self._log_reprioritization(event_type, changes)
\`\`\`

This responsive approach ensures agents maintain optimal focus even as operational conditions evolve.`
    },
    {
      title: 'Hierarchical Task Organization',
      content: `Effective prioritization operates at multiple organizational levels, from strategic goal selection to tactical action ordering, enabling comprehensive decision-making across complex agent architectures.

**Strategic Goal Prioritization**

High-level objective selection considering organizational priorities and resource constraints:

**Mission-Critical Objectives**: Core business functions requiring continuous attention
- System availability and performance maintenance
- Security and compliance requirements
- Revenue-generating activities and customer satisfaction

**Strategic Initiatives**: Long-term projects supporting competitive advantage
- Product development and innovation projects  
- Market expansion and partnership opportunities
- Technology modernization and capability enhancement

**Operational Excellence**: Supporting activities enabling efficient execution
- Process optimization and automation initiatives
- Team development and knowledge sharing
- Infrastructure improvement and maintenance

**Tactical Sub-Task Ordering**

Within strategic objectives, intelligent sequencing of constituent activities:

**Dependency-Driven Sequencing**: Topological sorting of prerequisite relationships
- Critical path identification for project timeline optimization
- Parallel task identification for resource utilization maximization
- Bottleneck detection and mitigation planning

**Resource-Optimal Scheduling**: Balancing workload across available capabilities
- Skill-based task assignment considering team member expertise
- Computational resource allocation for automated processes
- Time-based scheduling accounting for availability and deadlines

**Risk-Adjusted Ordering**: Prioritizing tasks that reduce project risk exposure
- Early validation of critical assumptions and technical feasibility
- Stakeholder alignment and approval acquisition
- External dependency resolution and contingency planning

**Immediate Action Selection**

Moment-to-moment decision-making for optimal agent behavior:

**Context-Aware Selection**: Current state consideration for next action determination
- Active task completion versus new task initiation
- Resource availability assessment for task feasibility
- User interaction requirements and response expectations

**Interruption Handling**: Graceful response to higher-priority events
- Current task state preservation for later resumption
- Priority comparison for interruption justification
- Context switching cost consideration

**Example Implementation:**

\`\`\`python
def get_next_task(self, worker_id: Optional[str] = None) -> Optional[AdvancedTask]:
    \"\"\"Get the highest priority task ready for execution.\"\"\"
    self._update_priority_queue()
    
    while self.task_queue:
        priority_score, task_id = heapq.heappop(self.task_queue)
        task = self.tasks.get(task_id)
        
        # Validate task readiness and worker assignment
        if (task and task.status == TaskStatus.PENDING and
            self._are_dependencies_satisfied(task) and
            self._check_resource_availability(task) and
            (not worker_id or not task.assigned_to or task.assigned_to == worker_id)):
            return task
    
    return None
\`\`\`

This hierarchical approach ensures coherent decision-making across all levels of agent operation.`
    },
    {
      title: 'LangChain Project Management Integration',
      content: `LangChain integration demonstrates practical implementation of intelligent prioritization through conversational interfaces and tool-based interactions.

**Agent-Based Task Management**

The LangChain framework enables natural language interaction with sophisticated prioritization logic:

**Conversational Task Creation**: Natural language processing for task specification
- Intent recognition for urgency and importance indicators
- Context extraction for effort estimation and resource requirements
- Worker assignment based on skill matching and availability

**Intelligent Priority Scoring**: Automatic evaluation based on conversation context
- Keyword analysis for urgency detection ("ASAP", "critical", "urgent")
- Business impact assessment from problem description
- Deadline extraction and time sensitivity calculation

**Dynamic Tool Selection**: Context-appropriate tool invocation for task management
- Task creation tools for new work item specification
- Priority update tools for changing circumstances
- Assignment tools for resource allocation optimization

**Agent Prompt Engineering**

Effective prompts guide the agent toward optimal prioritization decisions:

\`\`\`python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are an Intelligent Project Manager specializing in advanced task prioritization.

When creating tasks:
1. Analyze requests for urgency indicators (ASAP, urgent, critical, deadline)
2. Assess importance based on business impact or strategic value  
3. Estimate effort realistically based on task complexity
4. Assign to appropriate workers based on task type and availability

Priority Scoring Guide:
- Urgency: 0.9+ (Critical deadlines), 0.7+ (Time-sensitive), 0.5 (Normal), 0.3- (Flexible)
- Importance: 0.9+ (Strategic/Revenue), 0.7+ (Important features), 0.5 (Normal), 0.3- (Nice-to-have)

Always show the final prioritized task list after making changes.\"\"\"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
\`\`\`

**Tool Integration Architecture**

Comprehensive tool suite supporting all aspects of intelligent task management:

**Task Creation Tools**: Sophisticated task specification with intelligent defaults
- Multi-parameter task creation with priority calculation
- Dependency specification and validation
- Resource requirement estimation and allocation

**Priority Management Tools**: Dynamic adjustment capabilities
- Real-time priority recalculation based on new information
- Bulk priority updates for systematic changes
- Historical priority tracking for analysis and learning

**Assignment and Scheduling Tools**: Optimal resource allocation
- Skill-based assignment recommendation
- Workload balancing across team members
- Deadline-aware scheduling with conflict resolution

**Analytics and Reporting Tools**: Comprehensive system visibility
- Priority distribution analysis and trends
- Resource utilization optimization recommendations
- Performance metrics and improvement suggestions

**Example Usage Flow:**

\`\`\`python
# User request processing
async def process_request(self, request: str) -> str:
    \"\"\"Process project management request through intelligent agent.\"\"\"
    try:
        result = await self.agent_executor.ainvoke({"input": request})
        return result.get("output", "No response generated")
    except Exception as e:
        return f"Error: {str(e)}"

# Typical interaction
request = "Create urgent task to fix payment gateway bug for Alice"
response = await pm.process_request(request)
# Agent automatically: detects urgency, assigns high priority, schedules for Alice
\`\`\`

This integration demonstrates how sophisticated prioritization logic can be made accessible through natural, conversational interfaces.`
    }
  ],
  practicalApplications: [
    'Automated customer support systems prioritizing urgent system outage reports over routine password resets with VIP customer preferences',
    'Cloud computing resource management scheduling critical applications during peak demand while deferring batch jobs to off-peak hours',
    'Autonomous driving systems continuously prioritizing collision avoidance over lane discipline and fuel efficiency optimization',
    'Financial trading bots prioritizing trades based on market conditions, risk tolerance, profit margins, and real-time news analysis',
    'Project management systems prioritizing tasks by deadlines, dependencies, team availability, and strategic business importance',
    'Cybersecurity monitoring agents prioritizing alerts by threat severity, potential impact, and critical asset exposure for immediate response',
    'Personal assistant AIs managing calendar events, reminders, and notifications according to user importance, deadlines, and current context',
    'Software development workflows prioritizing bug fixes, feature development, and technical debt based on customer impact and resource availability'
  ],
  practicalExamples: [
    {
      title: 'Enterprise Project Management System',
      description: 'Comprehensive project management platform using intelligent prioritization for software development teams with multiple competing priorities and resource constraints.',
      implementation: 'LangChain-based conversational interface with multi-criteria decision framework, dynamic re-prioritization based on deadline proximity, resource-aware task scheduling, and team skill matching for optimal assignment allocation.'
    },
    {
      title: 'Customer Support Ticket Routing',
      description: 'Intelligent support system prioritizing customer issues based on severity, customer tier, business impact, and agent expertise for optimal resolution efficiency.',
      implementation: 'Real-time priority calculation considering SLA requirements, customer value scoring, issue complexity assessment, escalation path optimization, and agent workload balancing with dynamic re-routing capabilities.'
    },
    {
      title: 'Cloud Resource Orchestration',
      description: 'Sophisticated cloud infrastructure management system prioritizing workload allocation based on performance requirements, cost optimization, and availability constraints.',
      implementation: 'Multi-dimensional priority scoring with deadline urgency, resource efficiency analysis, cascading dependency management, and automatic failover prioritization with real-time cost optimization and performance monitoring.'
    }
  ],
  nextSteps: [
    'Implement basic multi-criteria prioritization frameworks incorporating urgency, importance, and resource efficiency in your agent systems',
    'Deploy dynamic re-prioritization mechanisms that respond to deadline changes, resource constraints, and external events automatically',
    'Establish hierarchical task organization supporting strategic goal selection, tactical sub-task ordering, and immediate action selection',
    'Create LangChain-based conversational interfaces for natural language task management and priority adjustment',
    'Integrate comprehensive analytics and reporting systems for prioritization effectiveness measurement and optimization',
    'Build dependency management systems handling blocking relationships, enabling dependencies, and circular dependency resolution',
    'Implement resource-aware scheduling algorithms optimizing allocation across computational, human, and external API resources',
    'Develop learning-based adaptation mechanisms that improve prioritization decisions based on historical performance and user feedback'
  ],
  references: [
    'Examining the Security of AI in Project Management: https://www.irejournals.com/paper-details/1706160',
    'AI-Driven Decision Support Systems in Agile Software Project Management: https://www.mdpi.com/2079-8954/13/3/208',
    'Multi-Criteria Decision Making in AI Systems: Academic Research Papers',
    'LangChain Framework Documentation: https://docs.langchain.com/',
    'Priority Queue Algorithms and Data Structures: Computer Science Fundamentals',
    'Resource Allocation Optimization in Distributed Systems: IEEE Papers',
    'Dynamic Task Scheduling in Real-Time Systems: ACM Digital Library'
  ],
  navigation: {
    previous: { href: '/chapters/evaluation-monitoring', title: 'Evaluation and Monitoring' },
    next: { href: '/chapters/pattern-discovery-innovation', title: 'Pattern Discovery and Innovation' }
  }
};
