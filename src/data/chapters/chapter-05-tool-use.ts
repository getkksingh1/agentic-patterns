import { Chapter } from '../types'

export const toolUseChapter: Chapter = {
  id: 'tool-use',
  number: 5,
  title: 'Tool Use',
  part: 'Part One – Foundations of Agentic Patterns',
  description: 'Enable agents to interact with external tools and APIs to extend their capabilities beyond text generation.',
  readingTime: '15 min read',
  difficulty: 'Intermediate',
  content: {
    overview: `Tool use is a fundamental pattern that extends agentic systems beyond pure text generation to interact with external tools, APIs, databases, and services. This pattern enables agents to perform concrete actions in the world, retrieve real-time information, and integrate with existing systems.

Effective tool use transforms agents from passive text generators into active problem-solvers that can accomplish real-world tasks by leveraging the appropriate tools for each situation.`,
    keyPoints: [
      'Define clear tool interfaces and capabilities',
      'Implement robust error handling for tool failures',
      'Manage tool authentication and permissions securely',
      'Optimize tool selection based on task requirements',
      'Monitor and log tool usage for analysis and debugging'
    ],
    codeExample: `# Example: Agent with multiple tool capabilities

class ToolUseAgent:
    def __init__(self):
        self.tools = {
            'web_search': WebSearchTool(),
            'calculator': CalculatorTool(),
            'email': EmailTool(),
            'calendar': CalendarTool(),
            'database': DatabaseTool()
        }
    
    def execute_with_tools(self, task):
        # Analyze task and determine required tools
        planning_prompt = f"""
        Task: {task}
        
        Available tools:
        - web_search: Search the internet for information
        - calculator: Perform mathematical calculations
        - email: Send emails to recipients
        - calendar: Check availability and schedule meetings
        - database: Query customer and product data
        
        Plan the steps needed and specify which tools to use:
        """
        
        plan = llm.complete(planning_prompt)
        
        # Execute plan step by step
        results = []
        for step in self.parse_plan(plan):
            if step['tool'] in self.tools:
                try:
                    result = self.tools[step['tool']].execute(step['parameters'])
                    results.append({
                        'step': step['description'],
                        'tool': step['tool'],
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'step': step['description'],
                        'tool': step['tool'],
                        'error': str(e)
                    })
        
        return self.synthesize_results(results)`,
    practicalApplications: [
      'Research assistance with web search and data analysis',
      'Customer service with CRM and knowledge base integration',
      'Financial analysis with market data and calculation tools',
      'Project management with scheduling and communication tools',
      'E-commerce with inventory and payment processing',
      'DevOps automation with deployment and monitoring tools'
    ],
    nextSteps: [
      'Start with simple, single-purpose tools',
      'Implement proper error handling and retries',
      'Add tool result validation and verification',
      'Explore tool composition and chaining',
      'Build a tool registry and discovery system'
    ]
  },
  navigation: {
    previous: { href: '/chapters/reflection', title: 'Reflection' },
    next: { href: '/chapters/planning', title: 'Planning' }
  }
}
