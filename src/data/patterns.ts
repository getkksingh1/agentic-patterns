export interface Pattern {
  id: string
  title: string
  description: string
  category: string
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  applications: string[]
  overview?: string
  implementation?: string[]
  examples?: string[]
}

export const patterns: Pattern[] = [
  {
    id: 'reasoning-chains',
    title: 'Reasoning Chains',
    description: 'Step-by-step logical reasoning for complex problem-solving tasks.',
    category: 'Reasoning',
    difficulty: 'Intermediate',
    applications: ['Problem Solving', 'Analysis', 'Decision Making'],
    overview: 'Reasoning chains enable agents to break down complex problems into smaller, manageable steps, following a logical sequence of thoughts to reach conclusions.',
    implementation: [
      'Define clear reasoning steps',
      'Implement step verification',
      'Handle chain interruptions',
      'Optimize for efficiency'
    ]
  },
  {
    id: 'tool-orchestration',
    title: 'Tool Orchestration',
    description: 'Coordinating multiple tools and APIs to accomplish complex tasks.',
    category: 'Implementation',
    difficulty: 'Advanced',
    applications: ['Automation', 'Integration', 'Workflow'],
    overview: 'Tool orchestration patterns help agents coordinate multiple external tools and services to accomplish tasks that require diverse capabilities.',
    implementation: [
      'Design tool registry',
      'Implement error handling',
      'Manage tool dependencies',
      'Monitor execution flow'
    ]
  },
  {
    id: 'memory-systems',
    title: 'Memory Systems',
    description: 'Persistent knowledge storage and retrieval for agent continuity.',
    category: 'Architecture',
    difficulty: 'Intermediate',
    applications: ['Learning', 'Context', 'Personalization'],
    overview: 'Memory systems provide agents with the ability to store, retrieve, and utilize information across sessions, enabling learning and context retention.'
  },
  {
    id: 'goal-decomposition',
    title: 'Goal Decomposition',
    description: 'Breaking down complex goals into achievable sub-goals and tasks.',
    category: 'Planning',
    difficulty: 'Intermediate',
    applications: ['Task Planning', 'Project Management', 'Strategy'],
    overview: 'Goal decomposition allows agents to tackle complex objectives by systematically breaking them into smaller, actionable components.'
  },
  {
    id: 'reactive-agents',
    title: 'Reactive Agents',
    description: 'Event-driven agents that respond to environmental changes in real-time.',
    category: 'Architecture',
    difficulty: 'Beginner',
    applications: ['Monitoring', 'Real-time Response', 'Event Processing'],
    overview: 'Reactive agents continuously monitor their environment and respond immediately to relevant changes or events.'
  },
  {
    id: 'planning-agents',
    title: 'Planning Agents',
    description: 'Deliberative agents that create and execute detailed plans to achieve goals.',
    category: 'Planning',
    difficulty: 'Advanced',
    applications: ['Strategy', 'Scheduling', 'Resource Management'],
    overview: 'Planning agents use sophisticated algorithms to create detailed plans before taking action, optimizing for efficiency and success.'
  },
  {
    id: 'multi-agent-coordination',
    title: 'Multi-Agent Coordination',
    description: 'Patterns for coordinating multiple agents working toward common goals.',
    category: 'Multi-Agent',
    difficulty: 'Advanced',
    applications: ['Team Coordination', 'Distributed Systems', 'Collaboration'],
    overview: 'Multi-agent coordination patterns enable multiple agents to work together effectively, sharing information and resources.'
  },
  {
    id: 'feedback-loops',
    title: 'Feedback Loops',
    description: 'Continuous learning and adaptation based on outcomes and feedback.',
    category: 'Learning',
    difficulty: 'Intermediate',
    applications: ['Adaptation', 'Optimization', 'Quality Improvement'],
    overview: 'Feedback loops allow agents to learn from their actions and continuously improve performance over time.'
  },
  {
    id: 'context-awareness',
    title: 'Context Awareness',
    description: 'Understanding and adapting to environmental and situational context.',
    category: 'Reasoning',
    difficulty: 'Intermediate',
    applications: ['Personalization', 'Adaptive Behavior', 'Situational Response'],
    overview: 'Context-aware agents understand their environment and situation to make more informed and appropriate decisions.'
  },
  {
    id: 'error-recovery',
    title: 'Error Recovery',
    description: 'Robust error handling and recovery mechanisms for autonomous operation.',
    category: 'Implementation',
    difficulty: 'Advanced',
    applications: ['Reliability', 'Fault Tolerance', 'System Recovery'],
    overview: 'Error recovery patterns ensure agents can handle failures gracefully and continue operating in the presence of errors.'
  },
  {
    id: 'delegation-patterns',
    title: 'Delegation Patterns',
    description: 'Strategies for delegating tasks to specialized agents or systems.',
    category: 'Multi-Agent',
    difficulty: 'Intermediate',
    applications: ['Task Distribution', 'Specialization', 'Load Balancing'],
    overview: 'Delegation patterns enable agents to efficiently distribute work to other agents or systems based on capabilities and availability.'
  },
  {
    id: 'observation-action',
    title: 'Observation-Action Cycles',
    description: 'Continuous perception and action cycles for dynamic environments.',
    category: 'Architecture',
    difficulty: 'Beginner',
    applications: ['Control Systems', 'Robotics', 'Interactive Systems'],
    overview: 'Observation-action cycles form the basic loop of autonomous agents, continuously perceiving and acting in their environment.'
  },
  {
    id: 'knowledge-graphs',
    title: 'Knowledge Graphs',
    description: 'Structured knowledge representation for enhanced reasoning capabilities.',
    category: 'Knowledge',
    difficulty: 'Advanced',
    applications: ['Knowledge Management', 'Reasoning', 'Information Retrieval'],
    overview: 'Knowledge graphs provide a structured way to represent and reason about complex relationships and information.'
  },
  {
    id: 'conversation-management',
    title: 'Conversation Management',
    description: 'Managing multi-turn conversations and dialogue state in interactive agents.',
    category: 'Interaction',
    difficulty: 'Intermediate',
    applications: ['Chatbots', 'Virtual Assistants', 'Customer Service'],
    overview: 'Conversation management patterns help agents maintain context and flow in extended interactions with users.'
  },
  {
    id: 'resource-allocation',
    title: 'Resource Allocation',
    description: 'Optimal allocation of computational and system resources among tasks.',
    category: 'Optimization',
    difficulty: 'Advanced',
    applications: ['Performance', 'Efficiency', 'Resource Management'],
    overview: 'Resource allocation patterns ensure agents use available resources efficiently while maintaining performance.'
  }
]

export const getPatternsByCategory = (category: string): Pattern[] => {
  return patterns.filter(pattern => pattern.category === category)
}

export const getPatternById = (id: string): Pattern | undefined => {
  return patterns.find(pattern => pattern.id === id)
}

export const getPatternsByDifficulty = (difficulty: string): Pattern[] => {
  return patterns.filter(pattern => pattern.difficulty === difficulty)
}
