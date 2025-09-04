import { Chapter } from '../types'

export const routingChapter: Chapter = {
  id: 'routing',
  number: 2,
  title: 'Routing',
  part: 'Part One – Foundations of Agentic Patterns',
  description: 'Master the art of dynamic decision-making that enables agents to adaptively select the most appropriate processing path based on context and intent.',
  readingTime: '20 min read',
  
  difficulty: 'Beginner',
  content: {
    overview: `While sequential processing via prompt chaining is foundational for executing deterministic workflows, real-world agentic systems must often arbitrate between multiple potential actions based on contingent factors. This capacity for dynamic decision-making, which governs the flow of control to different specialized functions, tools, or sub-processes, is achieved through routing.

Routing introduces conditional logic into an agent's operational framework, enabling a shift from fixed execution paths to a model where the agent dynamically evaluates specific criteria to select from a set of possible subsequent actions. This allows for more flexible and context-aware system behavior that can respond appropriately to a wider range of inputs and state changes.`,

    keyPoints: [
      'Enables dynamic decision-making based on content, context, and environmental state',
      'Transforms static execution paths into flexible, adaptive workflows',
      'Supports multiple routing mechanisms: LLM-based, embedding-based, rule-based, and ML model-based',
      'Acts as an intelligent dispatcher that matches user intent to appropriate handlers',
      'Essential for building sophisticated multi-agent and multi-tool systems',
      'Provides the foundation for conditional logic in agent architectures',
      'Enables agents to handle diverse inputs without predetermined response pathways'
    ],

    codeExample: `# LangChain Implementation: Coordinator Agent with Dynamic Routing
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# Initialize language model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# --- Define Specialized Handlers ---
def booking_handler(request: str) -> str:
    """Handles booking requests for flights and hotels."""
    print("\\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed: '{request}'. Result: Simulated booking action."

def info_handler(request: str) -> str:
    """Handles general information requests."""
    print("\\n--- DELEGATING TO INFO HANDLER ---")  
    return f"Info Handler processed: '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be classified."""
    print("\\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Request unclear: '{request}'. Please provide more specific details."

# --- Define Router Chain ---
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Analyze the user's request and determine the appropriate handler:
    
    - If related to booking flights or hotels, output 'booker'
    - For general information questions, output 'info'  
    - If unclear or ambiguous, output 'unclear'
    
    ONLY output one word: 'booker', 'info', or 'unclear'\"\"\"),
    ("user", "{request}")
])

coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# --- Define Routing Logic with RunnableBranch ---
branches = {
    "booker": RunnablePassthrough.assign(
        output=lambda x: booking_handler(x['request']['request'])
    ),
    "info": RunnablePassthrough.assign(
        output=lambda x: info_handler(x['request']['request'])
    ),
    "unclear": RunnablePassthrough.assign(
        output=lambda x: unclear_handler(x['request']['request'])
    ),
}

# Create routing branch logic
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip() == 'info', branches["info"]),
    branches["unclear"]  # Default branch
)

# --- Complete Coordinator Agent ---
coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output'])

# --- Example Usage ---
def run_examples():
    # Booking request
    result1 = coordinator_agent.invoke({"request": "Book me a flight to London"})
    print(f"Result 1: {result1}")
    
    # Information request  
    result2 = coordinator_agent.invoke({"request": "What is the capital of Italy?"})
    print(f"Result 2: {result2}")
    
    # Unclear request
    result3 = coordinator_agent.invoke({"request": "Help me with something"})
    print(f"Result 3: {result3}")

# Alternative Google ADK Implementation
# pip install google-adk google-genai

\"\"\"
import uuid
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

# Define specialized tool functions
def booking_tool_handler(request: str) -> str:
    return f"Booking action for '{request}' completed successfully."

def info_tool_handler(request: str) -> str:
    return f"Information retrieved for '{request}': Sample response."

# Create tools and specialized agents
booking_tool = FunctionTool(booking_tool_handler)
info_tool = FunctionTool(info_tool_handler)

booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="Handles flight and hotel booking requests",
    tools=[booking_tool]
)

info_agent = Agent(
    name="Info", 
    model="gemini-2.0-flash",
    description="Provides general information and answers questions",
    tools=[info_tool]
)

# Coordinator with automatic routing via sub_agents
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction="Analyze requests and delegate to appropriate specialist agents. Do not answer directly.",
    sub_agents=[booking_agent, info_agent]
)

# Usage with InMemoryRunner
async def run_adk_example():
    runner = InMemoryRunner(coordinator)
    result = await runner.run(user_id="user_123", session_id=str(uuid.uuid4()), 
                             new_message=types.Content(role='user', parts=[types.Part(text="Book a hotel in Paris")]))
    return result
\"\"\"`,

    practicalApplications: [
      '🎯 Intent Classification: Virtual assistants routing queries to appropriate skill modules',
      '📧 Content Triage: Email systems directing messages to sales, support, or technical teams', 
      '🔧 Tool Selection: Multi-tool agents choosing the right API or service for each task',
      '🎓 Educational Routing: AI tutors selecting curriculum modules based on student performance',
      '📊 Data Pipeline Routing: Processing systems directing data to appropriate transformation workflows',
      '🛡️ Escalation Management: Customer service bots routing complex issues to human agents',
      '🔍 Multi-Agent Coordination: Research systems assigning tasks to specialized analysis agents'
    ],

    nextSteps: [
      'Install required libraries: pip install langchain langgraph google-genai langchain-google-genai',
      'Set up API credentials for your chosen LLM provider (Google, OpenAI, Anthropic)',
      'Practice implementing basic intent classification with simple routing logic',
      'Experiment with embedding-based routing for semantic similarity matching',
      'Build rule-based fallback mechanisms for edge cases and error handling',
      'Explore advanced frameworks like LangGraph for complex state-based routing',
      'Study performance optimization and monitoring techniques for routing decisions'
    ]
  },

  sections: [
    {
      title: 'Four Types of Routing Mechanisms',
      content: `Routing can be implemented through several complementary approaches, each with distinct advantages:

**1. LLM-Based Routing**
The language model analyzes input and outputs a specific identifier indicating the next step. This approach leverages the model's natural language understanding capabilities.

*Example:* "Analyze this user query and output only the category: 'Order Status', 'Product Info', 'Technical Support', or 'Other'."

**2. Embedding-Based Routing**  
Input queries are converted into vector embeddings and compared to embeddings representing different routes. The query routes to the most semantically similar option.

*Best for:* Semantic routing where decisions are based on meaning rather than keywords.

**3. Rule-Based Routing**
Uses predefined logic (if-else statements, switch cases) based on keywords, patterns, or structured data extracted from input.

*Advantages:* Faster and more deterministic than LLM-based routing
*Limitations:* Less flexible for handling nuanced or novel inputs

**4. Machine Learning Model-Based Routing**
Employs a discriminative classifier trained on labeled data to perform routing decisions. The routing logic is encoded in the model's learned weights rather than executed via prompts.

*Key Characteristic:* Supervised fine-tuning creates specialized routing functions separate from generative models.`
    },
    {
      title: 'From Static to Dynamic Agent Behavior',
      content: `Traditional sequential processing follows a predetermined path regardless of input variation. Routing transforms this into adaptive behavior:

**Before Routing (Static):**
User Query → Fixed Processing Chain → Standard Response

**After Routing (Dynamic):**
User Query → Intent Analysis → Route Selection → Specialized Processing → Contextual Response

**Real-World Example: Customer Service Agent**

*Static Approach:* All queries follow the same response template
*Routing Approach:*
- Analyze query intent
- Route to appropriate specialist:
  - "Check order status" → Order Management System
  - "Product information" → Product Catalog Agent  
  - "Technical support" → Troubleshooting Agent
  - "Unclear intent" → Clarification Agent

This conditional logic enables agents to move beyond deterministic sequential processing and develop adaptive execution flows.`
    },
    {
      title: 'Implementation Strategies Across Frameworks',
      content: `Different frameworks provide various approaches to implementing routing logic:

**LangChain & LangGraph**
- Explicit constructs for defining conditional logic
- RunnableBranch for decision trees
- State-based graph architecture for complex routing scenarios
- Visual representation of routing flows

**Google Agent Development Kit (ADK)**
- Tool-based architecture where routing selects appropriate tools
- Auto-Flow mechanism for automatic delegation to sub-agents
- Instruction-based routing through agent descriptions
- Built-in intent matching and tool selection

**Implementation Locations:**
Routing can be applied at multiple points in an agent's cycle:
- **Initial Classification:** Determine primary task type
- **Intermediate Decision Points:** Select next action in processing chain
- **Tool Selection:** Choose appropriate API or service for current step
- **Escalation Logic:** Decide when to involve human operators`
    },
    {
      title: 'Beyond Simple Classification: Advanced Routing Patterns',
      content: `Sophisticated routing goes beyond basic intent classification:

**Multi-Criteria Routing**
Consider multiple factors simultaneously:
- Content analysis (what is being asked)
- User context (who is asking)
- System state (current load, availability)
- Urgency indicators (time sensitivity)
- Confidence levels (certainty of classification)

**Hierarchical Routing**
Implement nested routing decisions:
1. Primary category (Sales, Support, Technical)
2. Secondary classification (New Customer, Existing Account)
3. Tertiary routing (Specific product line, issue type)

**Adaptive Routing**
Learn and improve routing decisions over time:
- Monitor routing accuracy and user satisfaction
- Adjust routing criteria based on feedback
- Implement A/B testing for routing strategies
- Use reinforcement learning for optimization

**Fallback and Escalation Chains**
Handle routing failures gracefully:
- Confidence thresholds for routing decisions
- Multiple fallback options when primary routing fails
- Human escalation triggers for complex or ambiguous cases
- Logging and monitoring for continuous improvement`
    }
  ],

  practicalExamples: [
    {
      title: 'Multi-Domain Customer Service Router',
      description: 'Intelligent routing system for customer service handling multiple inquiry types',
      steps: [
        'Analyze incoming customer query for intent and urgency',
        'Extract key entities (account info, product names, issue type)',
        'Apply multi-criteria routing considering user tier and history',
        'Route to specialized agents: Sales, Technical Support, or Account Management',
        'Implement fallback to human agent for complex or ambiguous cases'
      ]
    },
    {
      title: 'Document Processing Pipeline Router',
      description: 'Automated system for routing different document types to appropriate processors',
      example: 'Invoice, Contract, Resume, or Legal Document uploaded to system',
      steps: [
        'Perform initial document classification using ML model',
        'Extract document metadata and structural features', 
        'Route to specialized processing workflows based on document type',
        'Apply document-specific validation and extraction rules',
        'Route processed results to appropriate downstream systems'
      ]
    },
    {
      title: 'Multi-Agent Research Coordinator',
      description: 'Research system coordinating multiple specialized AI agents',
      steps: [
        'Analyze research query to identify required capabilities',
        'Decompose complex queries into sub-tasks',
        'Route sub-tasks to appropriate specialist agents (Search, Analysis, Synthesis)',
        'Coordinate parallel execution and results aggregation',
        'Route final synthesis to presentation or report generation agent'
      ]
    }
  ],

  references: [
    'LangChain Documentation: https://www.langchain.com/',
    'LangGraph Documentation: https://langchain-ai.github.io/langgraph/',
    'Google Agent Developer Kit: https://google.github.io/adk-docs/',
    'RunnableBranch Guide: https://python.langchain.com/docs/expression_language/primitives/branching',
    'Intent Classification Techniques: https://www.promptingguide.ai/techniques/classification',
    'Embedding-Based Routing: https://python.langchain.com/docs/modules/data_connection/vectorstores/'
  ],

  navigation: {
    previous: { href: '/chapters/prompt-chaining', title: 'Prompt Chaining' },
    next: { href: '/chapters/parallelization', title: 'Parallelization' }
  }
}
