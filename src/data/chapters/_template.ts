import { Chapter } from '../types'

// Template for new chapter creation
// Copy this file and rename to chapter-XX-chapter-name.ts

export const templateChapter: Chapter = {
  id: 'chapter-slug', // URL-friendly identifier
  title: 'Chapter Title', // Display title
  subtitle: 'Chapter Subtitle (optional)', // Additional context
  description: 'Brief description of what this chapter covers and its importance in the context of agentic patterns',
  readingTime: '15 min read',
  
  // Main content - direct properties (new structure)
  overview: `Detailed overview explaining the pattern, its purpose, benefits, and how it fits into the broader agentic ecosystem. This should be 2-3 paragraphs providing context and setting expectations for the reader.

Include specific details about when and why to use this pattern, what problems it solves, and how it relates to other patterns in the guide.`,

  keyPoints: [
    'Key concept or benefit #1 that this pattern provides',
    'Important implementation detail or consideration #2', 
    'Critical insight or best practice #3',
    'Performance or scalability advantage #4',
    'Integration point or compatibility note #5'
  ],

  codeExample: `# Example Implementation of [Pattern Name]
# This should be a complete, runnable example that demonstrates the core concepts

import asyncio
from typing import Dict, Any, List, Optional

class ExampleAgent:
    """
    Example agent implementation demonstrating the pattern.
    Include docstrings explaining the key concepts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def example_method(self, input_data: str) -> Dict[str, Any]:
        """
        Main method demonstrating the pattern implementation.
        
        Args:
            input_data: Example input parameter
            
        Returns:
            Dictionary containing results and metadata
        """
        # Implementation steps with comments
        result = await self._process_input(input_data)
        return {
            "result": result,
            "metadata": {"pattern": "example-pattern"}
        }
    
    async def _process_input(self, data: str) -> str:
        """Helper method showing internal implementation."""
        return f"Processed: {data}"

# Usage example
async def demonstrate_pattern():
    agent = ExampleAgent({"param": "value"})
    result = await agent.example_method("test input")
    print(f"Result: {result}")

# Run demonstration
if __name__ == "__main__":
    asyncio.run(demonstrate_pattern())`,

  // Optional: Detailed sections
  sections: [
    {
      title: 'Core Concepts',
      content: `Detailed explanation of the fundamental concepts behind this pattern. Include technical details, architectural considerations, and theoretical background.

Multiple paragraphs explaining how the pattern works, why it's effective, and what makes it unique among other approaches.`
    },
    {
      title: 'Implementation Details',
      content: `Practical implementation guidance including:

• Setup and configuration requirements
• Step-by-step implementation process  
• Common pitfalls and how to avoid them
• Performance considerations
• Integration with existing systems`
    },
    {
      title: 'Advanced Techniques',
      content: `Advanced usage patterns and optimizations:

• Scaling strategies
• Error handling and resilience
• Monitoring and observability
• Security considerations
• Production deployment best practices`
    }
  ],

  practicalApplications: [
    'Use case #1: Specific scenario where this pattern excels',
    'Use case #2: Integration with existing systems or workflows',
    'Use case #3: Performance optimization or scalability scenario',
    'Use case #4: Error handling or reliability improvement',
    'Use case #5: Multi-agent coordination or collaboration'
  ],

  // Optional: Real-world examples
  practicalExamples: [
    {
      title: 'Example Scenario Title',
      description: 'Brief description of what this example demonstrates',
      implementation: 'Detailed implementation description or specific example text'
    }
  ],
  
  nextSteps: [
    'Start by implementing basic version with minimal configuration',
    'Add monitoring and logging to understand performance characteristics', 
    'Integrate with existing agent framework or system',
    'Test with real-world data and scenarios to validate effectiveness',
    'Scale implementation and optimize for production deployment'
  ],
  
  // Optional: Reference links
  references: [
    'Documentation Link: https://example.com/docs',
    'Research Paper: "Paper Title" by Authors (Year)',
    'Tutorial Resource: https://example.com/tutorial',
    'Framework Documentation: https://example.com/framework'
  ],
  
  // Navigation (update based on actual chapter position)
  navigation: {
    previous: { href: '/chapters/previous-chapter-slug', title: 'Previous Chapter Title' },
    next: { href: '/chapters/next-chapter-slug', title: 'Next Chapter Title' }
  }
}

// Steps to use this template:
// 1. Copy this file to chapter-XX-chapter-name.ts
// 2. Replace all placeholder content with actual chapter data
// 3. Update the navigation links based on chapter position
// 4. Add to index.ts exports
// 5. Add to chapters.ts array
// 6. Update README.md with completion status