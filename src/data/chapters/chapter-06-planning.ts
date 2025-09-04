import { Chapter } from '../types'

export const planningChapter: Chapter = {
  id: 'planning',
  number: 6,
  title: 'Planning',
  part: 'Part One – Foundations of Agentic Patterns',
  description: 'Enable agents to decompose complex goals into structured, sequential action plans for systematic goal achievement and workflow orchestration.',
  readingTime: '26 min read',
  difficulty: 'Intermediate',
  content: {
    overview: `Intelligent behavior extends beyond reacting to immediate input—it requires foresight, strategic thinking, and the ability to break down complex objectives into manageable, executable steps. The Planning pattern enables agents to function as strategic specialists who autonomously chart courses toward specified goals.

When you delegate a complex goal like "organize a team offsite" to a planning agent, you're defining the what—the objective and constraints—but not the how. The agent's core responsibility is to understand the initial state (budget, participants, dates) and goal state (successful offsite), then discover the optimal sequence of actions to connect them.

This pattern transforms agents from simple reactive systems into proactive strategists capable of handling multifaceted requests through structured decomposition, dependency management, and adaptive execution. The plan itself is not predetermined but emerges dynamically in response to the specific request and context.`,

    keyPoints: [
      'Decomposes high-level objectives into structured sequences of actionable sub-tasks and goals',
      'Enables autonomous strategy formulation for complex, multi-step workflows and processes',
      'Provides systematic dependency management and logical ordering of interdependent operations',
      'Supports dynamic adaptation when obstacles arise or new information becomes available',
      'Essential for workflow automation, research synthesis, and complex problem-solving scenarios',
      'Leverages LLM capabilities to generate plausible, effective plans based on extensive training data',
      'Balances flexibility for unknown solutions with predictability for well-understood processes',
      'Transforms reactive agents into strategic, goal-oriented systems capable of proactive execution'
    ],

    codeExample: `# CrewAI Implementation: Planning Agent for Research and Content Creation
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables for secure API key management
load_dotenv()

# Initialize language model with explicit configuration
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

# Define specialized planning and execution agent
planner_writer_agent = Agent(
    role='Strategic Content Planner and Writer',
    goal='Create comprehensive plans and execute structured content creation for complex topics.',
    backstory=(
        'You are an expert research strategist and technical writer with deep experience '
        'in breaking down complex subjects into manageable components. Your strength lies '
        'in creating detailed, actionable plans before execution, ensuring thorough coverage '
        'and logical flow in your final outputs. You excel at identifying key themes, '
        'structuring information hierarchically, and synthesizing insights from multiple perspectives.'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define complex planning task with structured requirements
def create_research_planning_task(topic: str, word_limit: int = 300):
    """
    Create a comprehensive research and writing task that requires planning.
    
    Args:
        topic: The research topic to investigate and write about
        word_limit: Maximum words for the final summary
    
    Returns:
        Task: Configured CrewAI task for planning and execution
    """
    
    return Task(
        description=(
            f"OBJECTIVE: Research and create a comprehensive summary on: '{topic}'\\n\\n"
            f"PROCESS REQUIREMENTS:\\n"
            f"1. PLANNING PHASE:\\n"
            f"   - Analyze the topic and identify 4-6 key subtopics or themes\\n"
            f"   - Determine the logical flow and structure for comprehensive coverage\\n"
            f"   - Identify potential knowledge gaps and areas requiring deeper investigation\\n"
            f"   - Create a detailed outline with main points and supporting details\\n\\n"
            f"2. RESEARCH PHASE:\\n"
            f"   - For each planned subtopic, consider multiple perspectives and approaches\\n"
            f"   - Identify the most current and relevant information available\\n"
            f"   - Note any conflicting viewpoints or emerging trends\\n\\n"
            f"3. WRITING PHASE:\\n"
            f"   - Execute the plan systematically, covering each planned element\\n"
            f"   - Ensure smooth transitions between planned sections\\n"
            f"   - Keep the summary to approximately {word_limit} words\\n"
            f"   - Include specific examples and concrete details where relevant"
        ),
        expected_output=(
            "A comprehensive report with the following structure:\\n\\n"
            "### RESEARCH PLAN\\n"
            "- Detailed bullet-point outline showing planned approach\\n"
            "- Key subtopics and their logical relationships\\n"
            "- Identified focus areas and research priorities\\n\\n"
            "### EXECUTIVE SUMMARY\\n"
            f"- Well-structured summary (≤{word_limit} words)\\n"
            "- Clear topic introduction and context\\n"
            "- Systematic coverage of all planned elements\\n"
            "- Concrete examples and specific insights\\n"
            "- Forward-looking conclusion with key implications\\n\\n"
            "### IMPLEMENTATION INSIGHTS\\n"
            "- Key challenges and opportunities identified\\n"
            "- Practical applications and next steps\\n"
            "- Areas requiring further investigation"
        ),
        agent=planner_writer_agent,
    )

# Advanced multi-task planning workflow
def create_multi_step_workflow(topics: list):
    """
    Create a multi-task workflow demonstrating complex planning scenarios.
    
    Args:
        topics: List of research topics to process sequentially
        
    Returns:
        tuple: (agents_list, tasks_list, crew)
    """
    
    # Create tasks for each topic
    tasks = []
    for i, topic in enumerate(topics, 1):
        task = create_research_planning_task(
            topic=topic, 
            word_limit=250
        )
        task.description = f"[TASK {i}/{len(topics)}] {task.description}"
        tasks.append(task)
    
    # Create crew with sequential processing
    crew = Crew(
        agents=[planner_writer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=True  # Enable memory across tasks for learning
    )
    
    return [planner_writer_agent], tasks, crew

# Execute planning workflow
if __name__ == "__main__":
    print("="*70)
    print("🎯 STRATEGIC PLANNING AGENT - RESEARCH WORKFLOW")
    print("="*70)
    
    # Single complex topic demonstration
    single_topic = "The role of artificial intelligence in sustainable urban development"
    print(f"\\n📋 SINGLE TOPIC ANALYSIS: {single_topic}")
    print("-" * 50)
    
    single_task = create_research_planning_task(single_topic, 350)
    single_crew = Crew(
        agents=[planner_writer_agent],
        tasks=[single_task],
        process=Process.sequential,
    )
    
    single_result = single_crew.kickoff()
    print("\\n📊 SINGLE TASK RESULT:")
    print(single_result)
    
    # Multi-topic workflow demonstration
    research_topics = [
        "Quantum computing applications in financial modeling",
        "Blockchain technology in supply chain transparency",
        "Machine learning in personalized healthcare"
    ]
    
    print(f"\\n\\n🔄 MULTI-TOPIC WORKFLOW: {len(research_topics)} Topics")
    print("-" * 50)
    
    agents, tasks, workflow_crew = create_multi_step_workflow(research_topics)
    
    print(f"Configured {len(agents)} agents and {len(tasks)} tasks")
    print("Executing sequential planning workflow...")
    
    try:
        workflow_result = workflow_crew.kickoff()
        print("\\n📊 WORKFLOW COMPLETION RESULT:")
        print(workflow_result)
        
    except Exception as e:
        print(f"❌ Workflow execution error: {e}")
        print("Please check your OpenAI API key and network connection.")

# Usage example:
# python planning_agent.py`,

    practicalApplications: [
      '📋 Business Process Automation: Employee onboarding workflows, project management sequences, and operational procedure execution',
      '🔬 Research & Analysis: Literature reviews, competitive analysis, market research with systematic information gathering and synthesis',
      '🤖 Autonomous Systems: Robotics navigation, path planning, and state-space traversal with obstacle avoidance and optimization',
      '📑 Content Creation: Multi-section report generation, documentation creation, and structured writing with logical flow',
      '🎯 Project Management: Task decomposition, dependency mapping, resource allocation, and milestone tracking for complex initiatives',
      '🏥 Healthcare Workflows: Treatment planning, diagnostic procedures, patient care coordination across multiple departments',
      '💼 Customer Service: Multi-step problem resolution, escalation procedures, and systematic issue diagnosis and resolution',
      '🌐 System Integration: API orchestration, data pipeline management, and complex workflow automation across multiple services'
    ],

    nextSteps: [
      'Install CrewAI and dependencies: pip install crewai langchain-openai python-dotenv',
      'Set up OpenAI API key in environment variables for secure access',
      'Practice creating simple planning agents that decompose basic multi-step tasks',
      'Experiment with different task complexity levels and planning granularity',
      'Study Google DeepResearch and OpenAI Deep Research for advanced planning examples',
      'Implement error handling and plan adaptation when initial steps fail',
      'Explore integration with Tool Use pattern for executable plan steps',
      'Build domain-specific planning templates for your particular use cases and workflows'
    ]
  },

  sections: [
    {
      title: 'Planning vs. Execution: The Strategic Decision Framework',
      content: `The decision to implement planning capabilities involves a critical trade-off between flexibility and predictability. Understanding when to use planning versus predetermined workflows is essential for effective system design.

**When Planning is Essential**
- **Unknown Solution Paths**: Problems where the "how" must be discovered rather than executed
- **Dynamic Environments**: Situations with changing constraints, unexpected obstacles, or evolving requirements
- **Complex Dependencies**: Multi-step processes where later steps depend on the outcomes of earlier ones
- **Goal-Oriented Tasks**: High-level objectives that can be achieved through multiple different approaches
- **Adaptive Requirements**: Scenarios requiring real-time adjustment based on intermediate results

**When Fixed Workflows are Preferable**
- **Well-Defined Procedures**: Processes with established, proven sequences that deliver consistent results
- **Compliance Requirements**: Operations that must follow specific regulatory or policy constraints
- **High-Risk Scenarios**: Critical systems where predictability and reliability take precedence over flexibility
- **Performance-Critical Applications**: Time-sensitive operations where planning overhead reduces efficiency
- **Simple Linear Tasks**: Straightforward sequences that don't benefit from dynamic adaptation

**The Strategic Decision Framework**
The key question driving this choice is: **"Does the 'how' need to be discovered, or is it already known?"**

- **Discovery Required** → Planning Pattern: Use when solution paths are unclear, constraints are dynamic, or creative problem-solving is needed
- **Execution Required** → Fixed Workflow: Use when procedures are established, compliance is critical, or performance demands predictability

**Hybrid Approaches**
Many sophisticated systems combine both approaches:
- **Hierarchical Planning**: High-level planning with fixed execution templates for common sub-tasks
- **Conditional Planning**: Predetermined workflows with planning triggers when exceptions occur
- **Adaptive Templates**: Flexible frameworks that can be customized through planning while maintaining structural consistency

This framework enables architects to make informed decisions about when planning adds value versus when it introduces unnecessary complexity.`
    },
    {
      title: 'Advanced Planning Systems: DeepResearch and Autonomous Investigation',
      content: `Modern planning systems have evolved beyond simple task decomposition to sophisticated autonomous research and investigation capabilities, exemplified by Google DeepResearch and OpenAI's Deep Research API.

**Google DeepResearch: Multi-Phase Autonomous Investigation**
Google's DeepResearch represents a paradigm shift in how AI systems approach complex research tasks:

- **Dynamic Plan Generation**: The system begins by deconstructing user prompts into multi-point research plans, presented for collaborative refinement
- **Iterative Search-Analysis Loop**: Rather than executing predefined searches, the agent dynamically formulates queries based on gathered information
- **Knowledge Gap Identification**: Active detection of information gaps, corroborating data points, and resolving discrepancies
- **Asynchronous Processing**: Long-running investigations that are resilient to failures and allow user disengagement
- **Synthesis and Structure**: Final outputs are structured, multi-page reports with integrated citations and interactive features

**OpenAI Deep Research API: Programmatic Research Automation**
The Deep Research API provides developers with direct access to sophisticated research planning capabilities:

- **Structured Input**: Define research parameters with explicit planning requirements and system messages
- **Model Selection**: Choose between o3-deep-research-2025-06-26 for quality or o4-mini-deep-research-2025-06-26 for speed
- **Tool Integration**: Enable web search, code execution, and custom MCP tools for comprehensive research
- **Reasoning Access**: Retrieve detailed planning phases and execution steps for transparency
- **Citation Management**: Automatic inline citation generation with source metadata and verification links

**Key Architectural Innovations**
1. **Transparent Planning**: Both systems expose their planning processes, allowing inspection of reasoning steps
2. **Tool Integration**: Seamless incorporation of web search, code execution, and data analysis tools
3. **Citation Management**: Automatic source tracking and inline citation generation
4. **Extensibility**: Support for custom tools and private knowledge base integration via Model Context Protocol (MCP)

**Performance Characteristics**
- **Comprehensiveness**: Processing hundreds of sources with systematic gap identification
- **Efficiency**: Automating time-intensive manual research cycles
- **Quality**: Structured outputs with verifiable citations and transparent methodology
- **Scalability**: Handling complex, multi-faceted research across diverse domains

These systems demonstrate how planning patterns can scale from simple task decomposition to sophisticated autonomous investigation, providing templates for building advanced research and analysis capabilities.`
    },
    {
      title: 'Implementation Patterns and Framework Integration',
      content: `Effective planning implementation requires careful consideration of framework capabilities, architectural patterns, and integration strategies across different use cases.

**CrewAI: Collaborative Planning Architecture**
CrewAI provides an excellent foundation for planning agents through its task-oriented design:

- **Multi-Agent Specialization**: Separate agents for strategic planning and execution coordination
- **Role-Based Design**: Agents with specific expertise areas (planning specialists, execution coordinators)
- **Task Dependencies**: Context-aware task execution where later tasks depend on planning outputs
- **Tool Integration**: Specialized tools assigned to appropriate agents based on their roles
- **Sequential Workflows**: Structured execution flow from planning through implementation

**LangGraph: State-Based Planning Workflows**
For more complex planning scenarios, LangGraph provides state management capabilities:

- **State Structure**: Define planning state with objectives, current plans, completed steps, and obstacles
- **Dynamic Adaptation**: Functions for plan creation, step execution, and replanning based on results
- **Conditional Logic**: Smart transitions between planning, execution, and adaptation phases
- **Error Handling**: Built-in mechanisms to detect failures and trigger replanning workflows
- **Graph Architecture**: Node-based workflow construction with edges defining execution paths

**Integration with Other Patterns**
Planning synergizes powerfully with other agentic patterns:

- **Planning + Tool Use**: Plans specify which tools to use at each step
- **Planning + Reflection**: Regular evaluation of plan effectiveness and adaptation
- **Planning + Memory**: Learning from past planning successes and failures
- **Planning + Routing**: Dynamic selection of execution paths based on context

**Best Practices for Planning Implementation**
1. **Clear Goal Definition**: Ensure objectives are specific and measurable
2. **Granular Step Decomposition**: Break complex tasks into atomic, executable actions
3. **Dependency Management**: Explicitly model relationships between plan steps
4. **Progress Tracking**: Implement mechanisms to monitor plan execution progress
5. **Adaptation Mechanisms**: Build in capabilities to modify plans when obstacles arise
6. **Resource Awareness**: Consider computational and time constraints in planning
7. **Validation Checkpoints**: Include verification steps to ensure plan quality before execution

These patterns and practices enable the construction of robust, scalable planning systems that can handle complex real-world scenarios while maintaining reliability and predictability.`
    }
  ],

  practicalExamples: [
    {
      title: 'Comprehensive Market Research Agent',
      description: 'Strategic research agent that plans and executes multi-phase competitive analysis with dynamic adaptation',
      example: 'Request: "Analyze the competitive landscape for AI-powered healthcare diagnostics, including key players, market trends, and regulatory considerations."',
      steps: [
        'PLANNING: Decompose into market segmentation, competitor identification, trend analysis, and regulatory research phases',
        'RESEARCH: Execute systematic information gathering using web search, industry reports, and patent databases',
        'ANALYSIS: Process collected data to identify patterns, competitive advantages, and market opportunities',
        'SYNTHESIS: Compile findings into structured report with executive summary, detailed analysis, and strategic recommendations',
        'ADAPTATION: Refine research focus based on emerging insights and identify areas requiring deeper investigation'
      ]
    },
    {
      title: 'Employee Onboarding Workflow Orchestrator',
      description: 'Business process automation agent that creates and manages complex onboarding sequences with dependency handling',
      steps: [
        'PLANNING: Map onboarding requirements including IT setup, HR documentation, training modules, and departmental introductions',
        'COORDINATION: Schedule interdependent tasks considering availability, prerequisites, and resource constraints',
        'EXECUTION: Trigger automated workflows for account creation, access provisioning, and training enrollment',
        'MONITORING: Track completion status, identify bottlenecks, and escalate delays or issues',
        'OPTIMIZATION: Analyze onboarding metrics and adapt process for improved efficiency and experience'
      ]
    },
    {
      title: 'Multi-Phase Product Development Planner',
      description: 'Project management agent that orchestrates complex development cycles with adaptive milestone management',
      example: 'Objective: "Plan and coordinate the development of a new mobile app feature from concept to launch."',
      steps: [
        'STRATEGIC PLANNING: Define project phases including research, design, development, testing, and launch preparation',
        'RESOURCE ALLOCATION: Assess team capacity, skill requirements, and timeline constraints for realistic scheduling',
        'DEPENDENCY MAPPING: Identify critical path dependencies and potential bottlenecks across development phases',
        'EXECUTION MONITORING: Track progress against milestones, manage scope changes, and coordinate cross-team communication',
        'ADAPTIVE MANAGEMENT: Adjust timelines and resource allocation based on progress updates and changing requirements'
      ]
    }
  ],

  references: [
    'Google DeepResearch (Gemini Feature): https://gemini.google.com',
    'OpenAI - Introducing Deep Research: https://openai.com/index/introducing-deep-research/',
    'Perplexity - Introducing Perplexity Deep Research: https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research',
    'CrewAI Documentation - Planning and Task Management: https://docs.crewai.com/concepts/tasks',
    'LangGraph - State-Based Workflows: https://langchain-ai.github.io/langgraph/',
    'Planning in AI Systems - MIT Course Materials: https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/'
  ],

  navigation: {
    previous: { href: '/chapters/tool-use', title: 'Tool Use' },
    next: { href: '/chapters/multi-agent', title: 'Multi-Agent' }
  }
}
