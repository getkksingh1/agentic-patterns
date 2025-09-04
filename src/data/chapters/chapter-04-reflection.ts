import { Chapter } from '../types'

export const reflectionChapter: Chapter = {
  id: 'reflection',
  number: 4,
  title: 'Reflection',
  part: 'Part One – Foundations of Agentic Patterns',
  description: 'Enable agents to iteratively evaluate and improve their own outputs through sophisticated self-correction and Producer-Critic feedback loops.',
  readingTime: '22 min read',
  difficulty: 'Intermediate',
  content: {
    overview: `Building upon the foundational patterns of Chaining, Routing, and Parallelization, the Reflection pattern addresses a critical limitation: even sophisticated agentic workflows can produce suboptimal, inaccurate, or incomplete initial outputs. Reflection introduces a metacognitive capability that enables agents to evaluate their own work and iteratively improve it.

The Reflection pattern involves an agent analyzing its own output, internal state, or reasoning process, then using that evaluation to refine its response. Unlike simple sequential chains that pass output directly to the next step, reflection introduces a feedback loop where the agent doesn't just produce output—it examines that output, identifies potential issues, and generates improved versions.

This pattern transforms agents from single-pass executors into self-aware systems capable of iterative refinement, leading to significantly higher quality outputs through a process of generation, evaluation, and improvement.`,

    keyPoints: [
      'Enables iterative self-correction and output refinement through feedback loops',
      'Implements Producer-Critic architecture for objective evaluation and improvement',
      'Provides metacognitive capabilities that analyze reasoning processes and outcomes',
      'Supports both self-reflection and separate critic agent evaluation mechanisms',
      'Significantly improves output quality at the cost of increased latency and computation',
      'Integrates with memory systems to learn from past critiques and avoid repeated errors',
      'Essential for high-stakes applications requiring accuracy, completeness, and adherence to complex constraints'
    ],

    codeExample: `# LangChain Implementation: Iterative Reflection Loop with Producer-Critic
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment and initialize LLM
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def run_reflection_loop():
    """
    Demonstrates iterative reflection with Producer-Critic architecture.
    Task: Create a robust Python factorial function with comprehensive requirements.
    """
    
    # --- Core Task Definition ---
    task_prompt = \"\"\"
    Create a Python function named 'calculate_factorial' that:
    1. Accepts a single integer 'n' as input
    2. Calculates its factorial (n!)
    3. Includes clear docstring explaining functionality
    4. Handles edge cases: factorial of 0 is 1
    5. Handles invalid input: raise ValueError for negative numbers
    \"\"\"
    
    # --- Reflection Loop Configuration ---
    max_iterations = 3
    current_code = ""
    message_history = [HumanMessage(content=task_prompt)]
    
    for i in range(max_iterations):
        print(f"\\n{'='*25} REFLECTION ITERATION {i + 1} {'='*25}")
        
        # --- PRODUCER STAGE: Generate or Refine Code ---
        if i == 0:
            print("\\n>>> PRODUCER: Generating initial code...")
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\\n>>> PRODUCER: Refining based on critique...")
            message_history.append(
                HumanMessage(content="Please refine the code using the critiques provided.")
            )
            response = llm.invoke(message_history)
            current_code = response.content
        
        print(f"\\n--- Generated Code (v{i + 1}) ---\\n{current_code}")
        message_history.append(response)
        
        # --- CRITIC STAGE: Evaluate Generated Code ---
        print("\\n>>> CRITIC: Evaluating code quality...")
        
        critic_prompt = [
            SystemMessage(content=\"\"\"
                You are a senior software engineer and Python expert.
                Your role is meticulous code review and quality assessment.
                
                Critically evaluate the provided code against requirements:
                - Check for bugs, style issues, and edge cases
                - Verify all requirements are met
                - Assess code clarity and documentation
                
                If code is perfect, respond with 'CODE_IS_PERFECT'.
                Otherwise, provide structured critique with specific improvements.
            \"\"\"),
            HumanMessage(content=f"Original Task:\\n{task_prompt}\\n\\nCode to Review:\\n{current_code}")
        ]
        
        critique_response = llm.invoke(critic_prompt)
        critique = critique_response.content
        
        # --- STOPPING CONDITION ---
        if "CODE_IS_PERFECT" in critique:
            print("\\n--- Critique ---\\nCode meets all requirements perfectly.")
            break
        
        print(f"\\n--- Critique ---\\n{critique}")
        message_history.append(
            HumanMessage(content=f"Critique of previous code:\\n{critique}")
        )
    
    print(f"\\n{'='*30} FINAL RESULT {'='*30}")
    print("\\nRefined code after reflection process:\\n")
    print(current_code)

# Usage: run_reflection_loop()

# Google ADK Alternative Implementation:
\"\"\"
from google.adk.agents import SequentialAgent, LlmAgent

# Producer Agent: Generates initial content
generator = LlmAgent(
    name="DraftWriter",
    description="Generates initial draft content on given subject.",
    instruction="Write a short, informative paragraph about the user's subject.",
    output_key="draft_text"
)

# Critic Agent: Evaluates and provides structured feedback
reviewer = LlmAgent(
    name="FactChecker", 
    description="Reviews text for factual accuracy with structured critique.",
    instruction=\"\"\"
    You are a meticulous fact-checker.
    1. Read text from state key 'draft_text'
    2. Verify factual accuracy of all claims
    3. Output dictionary with:
       - "status": "ACCURATE" or "INACCURATE"
       - "reasoning": Clear explanation with specific issues
    \"\"\",
    output_key="review_output"
)

# Sequential Pipeline: Producer → Critic workflow
review_pipeline = SequentialAgent(
    name="WriteAndReview_Pipeline",
    sub_agents=[generator, reviewer]
)

# Execution: generator saves to draft_text → reviewer evaluates → saves to review_output
\"\"\"`,

    practicalApplications: [
      '✍️ Creative Writing: Iterative refinement of stories, articles, and marketing copy for improved flow and engagement',
      '💻 Code Generation: Writing, testing, debugging, and optimizing code with automated error detection',
      '🧩 Complex Problem Solving: Multi-step reasoning with backtracking and solution path optimization',
      '📄 Document Summarization: Refining summaries for accuracy, completeness, and appropriate conciseness',
      '📋 Planning & Strategy: Evaluating proposed plans against constraints and improving feasibility',
      '💬 Conversational Agents: Reviewing dialogue history to maintain context and improve response quality',
      '🔬 Research & Analysis: Iterative hypothesis refinement and evidence validation in scientific contexts'
    ],

    nextSteps: [
      'Install required libraries: pip install langchain langchain-openai python-dotenv',
      'Set up environment variables (OPENAI_API_KEY) for model access',
      'Implement basic Producer-Critic architecture with single reflection cycle',
      'Experiment with different critic personas for specialized evaluation tasks',
      'Build iterative reflection loops with proper stopping conditions and max iterations',
      'Integrate reflection with memory systems to learn from past critiques',
      'Study cost-benefit trade-offs between reflection quality and computational expense'
    ]
  },

  sections: [
    {
      title: 'The Producer-Critic Architecture: Separation of Concerns',
      content: `A highly effective implementation of the Reflection pattern separates the process into two distinct logical roles, often called the "Generator-Critic" or "Producer-Reviewer" model.

**The Producer Agent**
- **Primary Role**: Performs initial task execution and content generation  
- **Focus**: Creating the first version of output (code, text, plans, analyses)
- **Optimization**: Dedicated entirely to generation quality without self-doubt
- **Input**: Takes initial prompt and user requirements
- **Output**: Produces first draft or attempt at solving the problem

**The Critic Agent**  
- **Primary Role**: Evaluates output generated by the Producer
- **Focus**: Finding flaws, suggesting improvements, providing structured feedback
- **Perspective**: Approaches output with fresh, unbiased evaluation
- **Criteria**: Analyzes against specific standards (accuracy, completeness, style, functionality)
- **Output**: Structured critique with actionable improvement recommendations

**Why Separation Works**
This architecture prevents the "cognitive bias" of self-review. The Critic approaches output with a dedicated analytical mindset, unconstrained by the Producer's original reasoning path. This separation enables:

- **Objectivity**: Fresh perspective unbiased by generation process
- **Specialization**: Each agent optimized for its specific role
- **Thoroughness**: Dedicated focus on finding issues vs. creating content
- **Scalability**: Different critic agents can provide domain-specific evaluation`
    },
    {
      title: 'Reflection vs. Other Agentic Patterns: Integration and Synergy',
      content: `Reflection doesn't operate in isolation—it integrates powerfully with other fundamental agentic patterns:

**Reflection + Memory (Chapter 8)**
Memory systems provide crucial context for evaluation, enabling:
- Learning from past critiques to avoid repeated errors
- Building cumulative improvement over multiple interactions  
- Context-aware refinement based on conversation history
- Progressive sophistication in self-evaluation capabilities

**Reflection + Goal Setting (Chapter 11)**
Goals provide the ultimate benchmark for self-evaluation:
- Reflection acts as the corrective engine using goal-oriented feedback
- Monitored progress informs the reflection process about deviations
- Adaptive strategy adjustment based on goal achievement analysis
- Transforms agents from passive executors to purposeful, adaptive systems

**Reflection + Tool Use (Chapter 5)**
External tools enhance the reflection process:
- Code execution and testing during code generation reflection
- Fact-checking tools for content accuracy verification  
- Performance measurement tools for optimization reflection
- External validation systems for objective quality assessment

**Implementation Synergy**
These patterns combine to create sophisticated agents that:
1. Generate initial outputs (Producer)
2. Evaluate against goals and past experience (Critic + Memory + Goals)
3. Use external tools for objective validation (Tool Use)
4. Iteratively refine until criteria are met (Reflection Loop)`
    },
    {
      title: 'Trade-offs and Implementation Considerations',
      content: `While the Reflection pattern significantly enhances output quality, it introduces important trade-offs that must be carefully considered:

**Cost and Latency Implications**
- **Multiple LLM Calls**: Each reflection cycle requires separate Producer and Critic calls
- **Increased Processing Time**: Iterative refinement can 2-5x execution time
- **API Rate Limits**: Higher call frequency may trigger service throttling  
- **Computational Overhead**: State management and conversation tracking

**Memory and Context Management**
- **Expanding Context**: Each iteration adds to conversation history
- **Context Window Limits**: Risk of exceeding model's maximum context length
- **Memory Intensive**: Storing multiple versions and critiques
- **State Complexity**: Managing Producer output, Critic feedback, and refinements

**Quality vs. Efficiency Balance**
- **Diminishing Returns**: Later iterations may provide minimal improvements
- **Optimal Stopping**: Determining when "good enough" has been achieved
- **Task Appropriateness**: Not all tasks benefit equally from reflection
- **Domain Expertise**: Some tasks require specialized evaluation criteria

**Best Practices for Implementation**
1. **Set Maximum Iterations**: Prevent infinite reflection loops
2. **Define Clear Success Criteria**: Explicit stopping conditions for "perfect" output  
3. **Monitor Context Length**: Implement truncation strategies for long conversations
4. **Cost Budgeting**: Set limits on reflection expense for different task types
5. **A/B Testing**: Compare reflection benefits against simpler approaches for specific use cases`
    },
    {
      title: 'Advanced Reflection Patterns and Techniques',
      content: `Beyond basic Producer-Critic architectures, sophisticated reflection implementations can leverage advanced patterns:

**Multi-Critic Evaluation**
- **Diverse Perspectives**: Multiple critics with different expertise areas
- **Consensus Building**: Aggregating feedback from multiple evaluation agents
- **Specialized Domains**: Technical accuracy, style, factual correctness evaluated separately
- **Weighted Feedback**: Prioritizing critiques based on critic expertise and task relevance

**Hierarchical Reflection**
- **Multi-Level Analysis**: Reflecting on both content and process
- **Meta-Reflection**: Evaluating the effectiveness of the reflection process itself
- **Strategic vs. Tactical**: High-level goal alignment and low-level implementation quality
- **Recursive Improvement**: Using reflection to improve reflection capabilities

**Dynamic Reflection Criteria**
- **Context-Adaptive**: Evaluation criteria that adjust based on task complexity
- **Progressive Standards**: Increasing quality thresholds as agent capabilities improve  
- **User-Defined Quality**: Incorporating user preferences into evaluation criteria
- **Domain-Specific Metrics**: Specialized evaluation for different problem domains

**Learning-Enhanced Reflection**
- **Pattern Recognition**: Identifying common failure modes from reflection history
- **Improvement Templates**: Reusable critique patterns for similar tasks
- **Personalization**: Adapting reflection style to user preferences and domain requirements
- **Continuous Calibration**: Adjusting reflection sensitivity based on downstream feedback`
    }
  ],

  practicalExamples: [
    {
      title: 'Code Generation with Automated Testing',
      description: 'Software development agent that writes, tests, and iteratively improves code quality',
      steps: [
        'Generate initial code implementation based on functional requirements',
        'Run automated tests and static analysis to identify bugs and style issues',
        'Critique code for edge cases, error handling, and documentation completeness',
        'Refine code based on test failures and critic feedback',
        'Repeat until all tests pass and code meets quality standards'
      ]
    },
    {
      title: 'Research Paper Writing Assistant',
      description: 'Academic writing agent that produces and refines scholarly content through multiple review cycles',
      example: 'Write a literature review on machine learning applications in healthcare',
      steps: [
        'Generate initial draft with literature synthesis and analysis',
        'Critique for academic rigor, citation accuracy, and logical flow',
        'Fact-check claims against source materials and identify gaps',
        'Refine argument structure and strengthen evidence presentation',
        'Final review for clarity, coherence, and publication readiness'
      ]
    },
    {
      title: 'Strategic Planning Agent with Feasibility Analysis',
      description: 'Business planning system that creates and optimizes strategic plans through iterative evaluation',
      steps: [
        'Generate initial strategic plan with goals, timelines, and resource allocation',
        'Evaluate plan against budget constraints, market conditions, and capabilities',
        'Identify potential risks, bottlenecks, and unrealistic assumptions',
        'Refine plan to address feasibility issues and optimize resource utilization',
        'Validate final plan against success criteria and stakeholder requirements'
      ]
    }
  ],

  references: [
    'Training Language Models to Self-Correct via Reinforcement Learning: https://arxiv.org/abs/2409.12917',
    'LangChain Expression Language (LCEL) Documentation: https://python.langchain.com/docs/introduction/',
    'LangGraph Documentation: https://www.langchain.com/langgraph',
    'Google Agent Developer Kit (ADK) Multi-Agent Systems: https://google.github.io/adk-docs/agents/multi-agents/',
    'Self-Refine: Iterative Refinement with Self-Feedback: https://arxiv.org/abs/2303.17651',
    'Constitutional AI: Harmlessness from AI Feedback: https://arxiv.org/abs/2212.08073'
  ],

  navigation: {
    previous: { href: '/chapters/parallelization', title: 'Parallelization' },
    next: { href: '/chapters/tool-use', title: 'Tool Use' }
  }
}
