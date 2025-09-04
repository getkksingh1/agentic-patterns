interface Appendix {
  id: string
  letter: string
  title: string
  description: string
  onlineOnly?: boolean
  content: {
    overview: string
    sections: {
      title: string
      content: string
    }[]
    resources?: string[]
  }
}

const appendices: Appendix[] = [
  {
    id: 'advanced-prompting',
    letter: 'A',
    title: 'Advanced Prompting Techniques',
    description: 'Deep dive into sophisticated prompting strategies for enhanced agent performance.',
    content: {
      overview: `This appendix explores advanced prompting techniques that go beyond basic instruction-following. These methods enable more sophisticated reasoning, better context utilization, and improved output quality in agentic systems.`,
      sections: [
        {
          title: 'Chain-of-Thought Prompting',
          content: `Chain-of-thought prompting encourages models to show their reasoning process step by step. This technique is particularly effective for complex problem-solving tasks.

Key principles:
• Explicitly ask for step-by-step reasoning
• Provide examples of good reasoning chains
• Use phrases like "Let's think step by step"
• Break down complex problems into smaller components

Example:
"To solve this math word problem, let's think step by step:
1. First, identify what we're trying to find
2. Then, extract the relevant numbers
3. Determine which operations to use
4. Calculate step by step
5. Check if the answer makes sense"`
        },
        {
          title: 'Few-Shot Learning',
          content: `Few-shot learning provides examples within the prompt to guide the model's behavior and output format.

Best practices:
• Use 2-5 high-quality examples
• Ensure examples cover edge cases
• Maintain consistent formatting
• Choose diverse but relevant examples

Structure:
Input: [example input 1]
Output: [example output 1]

Input: [example input 2]  
Output: [example output 2]

Input: [actual input]
Output: [model generates this]`
        },
        {
          title: 'Self-Consistency Techniques',
          content: `Self-consistency involves generating multiple responses and selecting the most consistent or confident answer.

Methods:
• Temperature sampling with multiple attempts
• Voting mechanisms for classification tasks
• Confidence scoring and selection
• Cross-validation of reasoning chains

This approach is particularly useful for:
• Critical decision-making tasks
• Mathematical problem solving
• Factual question answering
• Complex reasoning challenges`
        }
      ],
      resources: [
        'Research papers on prompting techniques',
        'OpenAI prompting best practices guide',
        'Anthropic\'s prompt engineering resources',
        'Google AI prompting documentation',
        'Community prompt libraries and examples'
      ]
    }
  },
  {
    id: 'agentic-frameworks',
    letter: 'C',
    title: 'Quick Overview of Agentic Frameworks',
    description: 'Survey of popular frameworks and tools for building agentic systems.',
    content: {
      overview: `This appendix provides a comprehensive overview of the most popular frameworks and tools available for building agentic systems. Each framework has its own strengths and is suited for different use cases and technical requirements.`,
      sections: [
        {
          title: 'LangChain',
          content: `LangChain is one of the most popular frameworks for building applications with large language models.

Strengths:
• Extensive ecosystem of integrations
• Strong community support
• Comprehensive documentation
• Supports multiple programming languages (Python, JavaScript)
• Rich set of pre-built components

Key Components:
• Chains: Sequential operations
• Agents: Autonomous decision-makers
• Tools: External integrations
• Memory: Conversation persistence
• Retrievers: Information retrieval

Best for: Rapid prototyping, educational projects, and applications requiring many integrations.`
        },
        {
          title: 'AutoGEN',
          content: `Microsoft's AutoGEN framework focuses on multi-agent conversations and collaboration.

Strengths:
• Multi-agent conversation patterns
• Built-in collaboration mechanisms
• Integration with Azure services
• Strong enterprise features
• Extensible agent types

Key Features:
• Conversational agents
• Group chat capabilities
• Code execution environments
• Human-in-the-loop integration
• Customizable agent behaviors

Best for: Enterprise applications, team-based AI systems, and complex multi-agent scenarios.`
        },
        {
          title: 'CrewAI',
          content: `CrewAI specializes in coordinated multi-agent systems with role-based collaboration.

Strengths:
• Role-based agent design
• Task delegation and coordination  
• Built-in collaboration patterns
• Simple configuration and setup
• Focus on business use cases

Core Concepts:
• Crews: Groups of agents working together
• Agents: Individual specialists with roles
• Tasks: Specific work assignments
• Tools: Capabilities agents can use
• Processes: Workflow orchestration

Best for: Business process automation, content creation teams, and structured collaborative tasks.`
        },
        {
          title: 'LlamaIndex',
          content: `LlamaIndex focuses on data ingestion, indexing, and retrieval for LLM applications.

Strengths:
• Advanced RAG capabilities
• Multiple data source connectors
• Sophisticated indexing strategies
• Query optimization
• Enterprise-ready features

Key Features:
• Data connectors for various sources
• Vector and keyword indexing
• Query engines and retrievers
• Response synthesis
• Evaluation frameworks

Best for: Knowledge-intensive applications, document analysis, and enterprise search.`
        }
      ],
      resources: [
        'Official framework documentation and tutorials',
        'Community examples and templates',
        'Performance benchmarks and comparisons',
        'Integration guides for popular services',
        'Best practices for framework selection'
      ]
    }
  }
]

export const getAppendixBySlug = (slug: string): Appendix | undefined => {
  return appendices.find(appendix => appendix.id === slug)
}

export const getAllAppendices = (): Appendix[] => {
  return appendices
}
