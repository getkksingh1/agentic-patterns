import { Chapter } from '../types'

export const promptChainingChapter: Chapter = {
  id: 'prompt-chaining',
  number: 1,
  title: 'Prompt Chaining',
  part: 'Part One – Foundations of Agentic Patterns',
  description: 'Master the foundational pattern of breaking complex tasks into sequential, manageable steps for more reliable and controlled AI interactions.',
  readingTime: '25 min read',
  difficulty: 'Beginner',
  content: {
    overview: `Prompt chaining, sometimes referred to as the Pipeline pattern, represents a powerful paradigm for handling intricate tasks when leveraging large language models (LLMs). Rather than expecting an LLM to solve a complex problem in a single, monolithic step, prompt chaining advocates for a divide-and-conquer strategy.

The core idea is to break down complex problems into a sequence of smaller, more manageable sub-problems. Each sub-problem is addressed individually through a specifically designed prompt, and the output generated from one prompt is strategically fed as input into the subsequent prompt in the chain.

This sequential processing technique introduces modularity and clarity into LLM interactions. By decomposing complex tasks, it becomes easier to understand, debug, and optimize each individual step, making the overall process more robust and interpretable.`,
    keyPoints: [
      'Transforms complex, monolithic tasks into manageable sequential steps',
      'Each step focuses on a specific aspect of the larger problem',
      'Output of one step serves as input for the next, creating dependency chains',
      'Enables integration of external knowledge, tools, and APIs at each step',
      'Provides foundation for sophisticated AI agents with multi-step reasoning',
      'Dramatically improves reliability and control over AI interactions',
      'Allows for better error handling and debugging at granular levels'
    ],
    codeExample: `# Complete LangChain Implementation Example
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Language Model
llm = ChatOpenAI(temperature=0)

# --- Prompt 1: Extract Information ---
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\\n\\n{text_input}"
)

# --- Prompt 2: Transform to JSON ---
prompt_transform = ChatPromptTemplate.from_template(
    """Transform the following specifications into a JSON object with 
    'cpu', 'memory', and 'storage' as keys:\\n\\n{specifications}"""
)

# --- Build the Chain using LCEL ---
extraction_chain = prompt_extract | llm | StrOutputParser()

# The full chain passes extraction output to transformation
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

# --- Execute the Chain ---
input_text = """The new laptop model features a 3.5 GHz octa-core processor, 
16GB of RAM, and a 1TB NVMe SSD."""

final_result = full_chain.invoke({"text_input": input_text})
print("Final JSON Output:", final_result)

# --- Advanced Multi-Step Analysis Chain ---
def create_analysis_chain():
    # Step 1: Market Research Summarization
    summarize_prompt = ChatPromptTemplate.from_template(
        """As a Market Analyst, summarize the key findings of this market research report:
        {report_text}
        
        Focus on the most important insights and data points."""
    )
    
    # Step 2: Trend Identification  
    trends_prompt = ChatPromptTemplate.from_template(
        """As a Trade Analyst, using this summary, identify the top 3 emerging trends.
        Extract specific data points that support each trend:
        
        {summary}
        
        Format as JSON with trend_name and supporting_data fields."""
    )
    
    # Step 3: Email Composition
    email_prompt = ChatPromptTemplate.from_template(
        """As an Expert Documentation Writer, draft a concise email to the marketing team 
        that outlines these trends and their supporting data:
        
        {trends}
        
        Use a professional tone and include actionable insights."""
    )
    
    # Create the chain
    summary_chain = summarize_prompt | llm | StrOutputParser()
    trends_chain = trends_prompt | llm | StrOutputParser()  
    email_chain = email_prompt | llm | StrOutputParser()
    
    # Complete analysis pipeline
    full_analysis_chain = (
        {"summary": summary_chain}
        | {"trends": trends_chain}
        | {"email": email_chain}
    )
    
    return full_analysis_chain`,
    practicalApplications: [
      '🔍 Information Processing Workflows: Multi-step document analysis, entity extraction, and report generation',
      '❓ Complex Query Answering: Breaking down complex questions into researchable sub-components',
      '📊 Data Extraction and Transformation: Converting unstructured text into validated, structured formats',
      '✍️ Content Generation Workflows: Ideation → Outlining → Drafting → Revision pipelines',
      '💬 Conversational Agents with State: Maintaining context across multi-turn dialogues',
      '⚡ Code Generation and Refinement: Problem analysis → Pseudocode → Implementation → Testing',
      '🎯 Multimodal Reasoning: Processing images with text, labels, and tabular data sequentially'
    ],
    nextSteps: [
      'Install required libraries: pip install langchain langchain-openai langgraph',
      'Set up API credentials for your chosen LLM provider (OpenAI, Anthropic, etc.)',
      'Practice implementing basic 2-3 step chains with structured outputs',
      'Experiment with error handling and retry mechanisms between steps',
      'Learn to validate outputs at each stage before passing to the next',
      'Explore advanced frameworks like LangGraph for stateful, cyclical workflows',
      'Study performance optimization techniques and cost management strategies'
    ]
  },
  sections: [
    {
      title: 'Why Single Prompts Fail for Complex Tasks',
      content: `For multifaceted tasks, using a single, complex prompt can be inefficient and unreliable. Common issues include:

**Instruction Neglect**: Parts of complex prompts are overlooked or ignored
**Contextual Drift**: The model loses track of initial context as the prompt progresses  
**Error Propagation**: Early errors amplify throughout the response
**Context Window Limitations**: Insufficient information to respond comprehensively
**Increased Hallucination**: Higher cognitive load increases chances of incorrect information

**Example of Problematic Single Prompt:**
"Analyze this market research report, summarize findings, identify trends with data points, and draft an email to the marketing team."

This request risks failure because the model might excel at summarization but fail to extract specific data points or properly format the email.`
    },
    {
      title: 'Enhanced Reliability Through Sequential Decomposition', 
      content: `Prompt chaining addresses single-prompt limitations by creating focused, sequential workflows:

**Step 1 - Initial Prompt (Summarization):**
"Summarize the key findings of the following market research report: [text]."
- Model focuses solely on summarization
- Increases accuracy of this foundational step

**Step 2 - Second Prompt (Trend Identification):**  
"Using the summary, identify the top three emerging trends and extract specific data points: [output from step 1]."
- More constrained scope builds on validated output
- Reduces ambiguity and cognitive load

**Step 3 - Third Prompt (Email Composition):**
"Draft a concise email to the marketing team outlining these trends: [output from step 2]."
- Final step focuses purely on communication
- Benefits from structured, verified input data

This decomposition provides granular control over each step, similar to computational pipelines where each function performs a specific operation before passing results to the next.`
    },
    {
      title: 'The Critical Role of Structured Output',
      content: `The reliability of prompt chains depends heavily on data integrity between steps. Ambiguous or poorly formatted output from one prompt can cause subsequent prompts to fail.

**Solution: Specify Structured Output Formats**
Use JSON, XML, or other machine-readable formats to ensure precise data transfer.

**Example Structured Output:**
\`\`\`json
{
  "trends": [
    {
      "trend_name": "AI-Powered Personalization",
      "supporting_data": "73% of consumers prefer brands using personal information for relevant shopping experiences."
    },
    {
      "trend_name": "Sustainable and Ethical Brands", 
      "supporting_data": "ESG-related products grew 28% vs 20% for products without ESG claims."
    }
  ]
}
\`\`\`

This structured approach:
- Ensures machine-readable output
- Eliminates interpretation ambiguity  
- Minimizes errors from natural language parsing
- Enables robust, multi-step LLM-based systems`
    },
    {
      title: 'Context Engineering vs. Traditional Prompt Engineering',
      content: `**Context Engineering** represents a significant evolution from traditional prompt engineering. Rather than just optimizing query phrasing, it focuses on building comprehensive informational environments for AI models.

**Components of Context Engineering:**

1. **System Prompts**: Foundational instructions defining operational parameters
   - "You are a technical writer; your tone must be formal and precise"

2. **External Data Integration**:
   - Retrieved documents from knowledge bases
   - Real-time tool outputs (APIs, databases, calculators)
   - Environmental state and user context

3. **Implicit Data**:
   - User identity and interaction history
   - Previous conversation context
   - Situational awareness

**The Engineering Approach:**
- Create robust data fetching pipelines
- Implement runtime data transformation
- Establish continuous feedback loops
- Use tools like Google's Vertex AI prompt optimizer for systematic improvement

This transforms simple chatbots into sophisticated, contextually-aware systems that understand not just what to do, but when and how to do it based on complete operational awareness.`
    }
  ],
  practicalExamples: [
    {
      title: 'Market Research Analysis Pipeline',
      description: 'Complete workflow for processing research reports into actionable insights',
      implementation: 'Sequential pipeline: text extraction → summarization → entity extraction → knowledge base querying → executive summary generation with structured data flow between stages'
    },
    {
      title: 'Complex Query Decomposition',
      description: 'Breaking down multi-faceted questions into researchable components',
      example: '"What were the main causes of the 1929 stock market crash, and how did government policy respond?"',
      steps: [
        'Identify core sub-questions (causes, government response)',
        'Research information about crash causes',  
        'Research government policy responses',
        'Synthesize information into coherent answer'
      ]
    },
    {
      title: 'OCR and Data Processing',
      description: 'Converting scanned documents into structured, validated data',
      steps: [
        'Extract raw text from document image',
        'Normalize data (convert text numbers to digits)',
        'Delegate calculations to external tools',
        'Validate and structure final results'
      ]
    }
  ],
  references: [
    'LangChain Documentation on LCEL: https://python.langchain.com/v0.2/docs/core_modules/expression_language/',
    'LangGraph Documentation: https://langchain-ai.github.io/langgraph/',
    'Prompt Engineering Guide - Chaining Prompts: https://www.promptingguide.ai/techniques/chaining',
    'OpenAI API Documentation: https://platform.openai.com/docs/guides/gpt/prompting',
    'Crew AI Documentation: https://docs.crewai.com/',
    'Google AI for Developers: https://cloud.google.com/discover/what-is-prompt-engineering',
    'Vertex Prompt Optimizer: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer'
  ],
  navigation: {
    next: { href: '/chapters/routing', title: 'Routing' }
  }
}
