import { Chapter } from '../types'

export const parallelizationChapter: Chapter = {
  id: 'parallelization',
  number: 3,
  title: 'Parallelization',
  part: 'Part One – Foundations of Agentic Patterns',
  description: 'Master concurrent execution of independent tasks to dramatically reduce latency and improve system responsiveness in complex agentic workflows.',
  readingTime: '18 min read',
  difficulty: 'Intermediate',
  content: {
    overview: `While Prompt Chaining handles sequential workflows and Routing enables dynamic decision-making, many complex agentic tasks involve multiple sub-tasks that can be executed simultaneously rather than one after another. This is where the Parallelization pattern becomes crucial.

Parallelization involves executing multiple components—such as LLM calls, tool usages, or entire sub-agents—concurrently. Instead of waiting for one step to complete before starting the next, parallel execution allows independent tasks to run simultaneously, significantly reducing overall execution time.

The core principle is identifying parts of the workflow that do not depend on each other's outputs and executing them in parallel. This is particularly effective when dealing with external services with latency, as you can issue multiple requests concurrently while the system waits.`,

    keyPoints: [
      'Execute independent tasks simultaneously to minimize total workflow time',
      'Identify workflow components that can run concurrently vs. sequentially', 
      'Leverage asynchronous execution and concurrency frameworks effectively',
      'Handle partial failures and implement robust error handling strategies',
      'Optimize resource utilization and manage concurrent operations efficiently',
      'Combine parallel execution with sequential synthesis for complex workflows',
      'Design fault-tolerant systems that gracefully handle concurrent operation failures'
    ],

    codeExample: `# LangChain Implementation: Parallel Processing with Synthesis
import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# Initialize language model
try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

# --- Define Independent Chains for Parallel Execution ---
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the following topic concisely:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Generate three interesting questions about the following topic:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

# --- Build Parallel Execution Block ---
# RunnableParallel executes all chains simultaneously
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),  # Pass original topic through
    }
)

# --- Define Synthesis Step ---
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Based on the following parallel processing results:
    Summary: {summary}
    Related Questions: {questions}
    Key Terms: {key_terms}
    
    Synthesize a comprehensive analysis that integrates all components.\"\"\"),
    ("user", "Original topic: {topic}")
])

# --- Complete Parallel + Sequential Pipeline ---
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- Execution Function ---
async def run_parallel_example(topic: str) -> None:
    \"\"\"Execute parallel processing workflow with topic synthesis.\"\"\"
    if not llm:
        print("LLM not initialized. Cannot run example.")
        return

    print(f"\\n--- Parallel Processing: '{topic}' ---")
    try:
        # Parallel execution: summary, questions, and terms run simultaneously
        response = await full_parallel_chain.ainvoke(topic)
        print("\\n--- Synthesized Result ---")
        print(response)
    except Exception as e:
        print(f"\\nError during parallel execution: {e}")

# --- Usage Example ---
# asyncio.run(run_parallel_example("The history of space exploration"))

# Google ADK Alternative Implementation:
\"\"\"
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

GEMINI_MODEL = "gemini-2.0-flash"

# Define parallel research agents
researcher_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="Research renewable energy sources. Summarize key findings.",
    tools=[google_search],
    output_key="renewable_energy_result"
)

researcher_2 = LlmAgent(
    name="EVResearcher", 
    model=GEMINI_MODEL,
    instruction="Research electric vehicle technology. Summarize developments.",
    tools=[google_search],
    output_key="ev_technology_result"
)

# Create parallel execution agent
parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_1, researcher_2],
    description="Runs multiple research agents concurrently."
)

# Synthesis agent (runs after parallel completion)
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction="Combine research findings: {renewable_energy_result}, {ev_technology_result}",
    description="Synthesizes parallel research results into structured report."
)

# Main sequential pipeline
sequential_pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesis."
)
\"\"\"`,

    practicalApplications: [
      '🔍 Information Gathering: Research agent collecting data from multiple sources simultaneously',
      '📊 Data Processing: Applying different analysis techniques concurrently across data segments',
      '🔧 Multi-API Integration: Travel agent checking flights, hotels, events, and restaurants in parallel',
      '✍️ Content Generation: Creating email components (subject, body, images, CTA) simultaneously',
      '✅ Validation Systems: Concurrent input verification (email format, phone, address, profanity)',
      '🎭 Multi-Modal Processing: Analyzing text sentiment and image content of social media posts concurrently',
      '🎯 A/B Testing: Generating multiple creative variations simultaneously for comparison and selection'
    ],

    nextSteps: [
      'Install required libraries: pip install langchain langchain-openai asyncio',
      'Practice identifying independent vs. dependent tasks in your workflows',
      'Implement basic parallel execution with RunnableParallel in LangChain',
      'Learn async/await patterns and concurrency best practices',
      'Build error handling and retry mechanisms for concurrent operations', 
      'Explore Google ADK ParallelAgent and SequentialAgent combinations',
      'Study performance optimization techniques for parallel agent systems'
    ]
  },

  sections: [
    {
      title: 'Sequential vs. Parallel Execution: A Practical Comparison',
      content: `Understanding when to use parallel vs. sequential execution is crucial for optimizing agent performance:

**Sequential Approach Example: Research Agent**
1. Search for Source A
2. Summarize Source A  
3. Search for Source B
4. Summarize Source B
5. Synthesize final answer from summaries A and B

*Total Time:* Sum of all individual task durations

**Parallel Approach Example: Same Research Agent**  
1. **Parallel Phase**: Search for Source A AND Search for Source B simultaneously
2. **Parallel Phase**: Summarize Source A AND Summarize Source B simultaneously  
3. **Sequential Phase**: Synthesize final answer (waits for parallel completion)

*Total Time:* Dramatically reduced due to concurrent operations

**Key Insight**: The synthesis step remains sequential because it depends on the parallel results, but the independent operations (searching and summarizing) can run concurrently.`
    },
    {
      title: 'Framework-Specific Implementation Strategies',
      content: `Different frameworks provide distinct mechanisms for implementing parallelization:

**LangChain Expression Language (LCEL)**
- **RunnableParallel**: Core construct for concurrent execution
- **Dictionary Structure**: Define multiple runnables that execute simultaneously
- **Result Aggregation**: Automatically collects outputs from parallel operations
- **Integration**: Seamlessly pipes into sequential synthesis steps

**Google Agent Development Kit (ADK)**
- **ParallelAgent**: Orchestrates concurrent execution of sub-agents
- **State Management**: Results stored in session state with output_key
- **SequentialAgent**: Combines parallel and sequential phases
- **Multi-Agent Coordination**: Native support for complex agent workflows

**Implementation Pattern:**
1. **Identify Independent Tasks**: Operations that don't depend on each other
2. **Define Parallel Block**: Group independent tasks for concurrent execution  
3. **Handle Results**: Aggregate outputs from parallel operations
4. **Sequential Synthesis**: Combine results in final processing step`
    },
    {
      title: 'Error Handling and Fault Tolerance in Parallel Systems',
      content: `Parallel execution introduces complexity that requires robust error handling:

**Common Failure Scenarios:**
- **Partial Failures**: Some parallel tasks succeed while others fail
- **Timeout Issues**: External API calls that exceed time limits
- **Resource Contention**: Multiple tasks competing for limited resources
- **Network Failures**: Intermittent connectivity affecting concurrent operations

**Fault Tolerance Strategies:**

**1. Graceful Degradation**
- Continue with successful results even if some parallel tasks fail
- Implement fallback mechanisms for failed operations
- Provide partial results with clear indication of missing components

**2. Retry Mechanisms**
- Exponential backoff for failed parallel operations
- Circuit breaker patterns for unreliable external services
- Selective retry based on failure type and criticality

**3. Timeout Management**
- Set appropriate timeouts for each parallel operation
- Implement overall workflow timeouts to prevent indefinite hanging
- Balance between giving operations enough time and maintaining responsiveness

**4. Result Validation**
- Validate outputs from each parallel operation before synthesis
- Handle malformed or unexpected results gracefully
- Implement quality checks for critical parallel processes`
    },
    {
      title: 'Performance Optimization and Resource Management',
      content: `Effective parallelization requires careful consideration of system resources and performance characteristics:

**Resource Management Considerations:**

**1. Concurrency Limits**
- Set appropriate limits on simultaneous operations
- Consider API rate limits and connection pools
- Balance parallelism with system resource constraints

**2. Memory Management**
- Monitor memory usage during parallel operations
- Implement streaming for large data processing
- Clean up resources promptly after parallel task completion

**3. CPU vs. I/O Bound Tasks**
- **I/O Bound**: Network requests, file operations (high parallelism benefit)
- **CPU Bound**: Computational tasks (limited by processor cores)
- **Mixed Workloads**: Balance different task types appropriately

**Performance Optimization Techniques:**

**1. Batching Strategies**
- Group related operations for efficient parallel execution
- Optimize batch sizes based on resource availability
- Implement dynamic batching based on system load

**2. Caching and Memoization**
- Cache results from expensive parallel operations
- Share cached results across parallel tasks when appropriate
- Implement intelligent cache invalidation strategies

**3. Monitoring and Metrics**
- Track parallel operation success rates and timing
- Monitor resource utilization during concurrent execution
- Implement alerting for performance degradation or failures`
    }
  ],

  practicalExamples: [
    {
      title: 'Multi-Source Research Agent',
      description: 'Intelligent research system gathering information from multiple sources concurrently',
      steps: [
        'Identify independent research sources (news, academic, social media, databases)',
        'Launch parallel search operations across all identified sources',
        'Apply concurrent analysis techniques (sentiment, keyword extraction, categorization)',
        'Handle partial failures gracefully while continuing with successful sources',
        'Synthesize findings from all parallel operations into comprehensive report'
      ]
    },
    {
      title: 'Travel Planning Optimization Engine',
      description: 'Comprehensive travel agent processing multiple booking options simultaneously',
      example: 'Plan a business trip to San Francisco with accommodation, transportation, and activities',
      steps: [
        'Launch parallel searches: flights, hotels, car rentals, restaurant recommendations',
        'Concurrently check availability and pricing across multiple providers',
        'Apply parallel validation: budget constraints, schedule compatibility, preferences',
        'Aggregate all options and apply multi-criteria optimization',
        'Present integrated travel plan with alternatives and booking links'
      ]
    },
    {
      title: 'Content Creation Pipeline with Multi-Modal Processing',
      description: 'Marketing content generator creating multiple components simultaneously',
      steps: [
        'Generate parallel content variations: headlines, body copy, call-to-action text',
        'Concurrently create visual elements: image selection, graphic design, layout options',
        'Apply parallel quality checks: brand compliance, tone analysis, readability',
        'Perform A/B testing preparation with multiple creative variations',
        'Synthesize final content package with optimization recommendations'
      ]
    }
  ],

  references: [
    'LangChain Expression Language (LCEL) Parallelism: https://python.langchain.com/docs/concepts/lcel/',
    'Google Agent Developer Kit Multi-Agent Systems: https://google.github.io/adk-docs/agents/multi-agents/',
    'Python asyncio Documentation: https://docs.python.org/3/library/asyncio.html',
    'RunnableParallel Guide: https://python.langchain.com/docs/expression_language/primitives/parallel',
    'Concurrency Best Practices: https://docs.python.org/3/library/concurrent.futures.html',
    'Asynchronous Programming Patterns: https://python.langchain.com/docs/expression_language/streaming'
  ],

  navigation: {
    previous: { href: '/chapters/routing', title: 'Routing' },
    next: { href: '/chapters/reflection', title: 'Reflection' }
  }
}
