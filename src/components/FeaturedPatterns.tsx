import Link from 'next/link'
import PatternCard from './PatternCard'

const featuredPatterns = [
  {
    id: 'reasoning-chains',
    title: 'Reasoning Chains',
    description: 'Step-by-step logical reasoning for complex problem-solving tasks.',
    category: 'Reasoning',
    difficulty: 'Intermediate',
    applications: ['Problem Solving', 'Analysis', 'Decision Making'],
  },
  {
    id: 'tool-orchestration',
    title: 'Tool Orchestration',
    description: 'Coordinating multiple tools and APIs to accomplish complex tasks.',
    category: 'Implementation',
    difficulty: 'Advanced',
    applications: ['Automation', 'Integration', 'Workflow'],
  },
  {
    id: 'memory-systems',
    title: 'Memory Systems',
    description: 'Persistent knowledge storage and retrieval for agent continuity.',
    category: 'Architecture',
    difficulty: 'Intermediate',
    applications: ['Learning', 'Context', 'Personalization'],
  },
]

export default function FeaturedPatterns() {
  return (
    <section className="section bg-gray-50">
      <div className="container">
        <div className="text-center mb-12">
          <h2 className="text-3xl lg:text-4xl font-bold mb-4">
            Featured Patterns
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Start with these essential patterns that form the foundation of effective agentic systems.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-8">
          {featuredPatterns.map((pattern) => (
            <PatternCard key={pattern.id} pattern={pattern} />
          ))}
        </div>

        <div className="text-center">
          <Link href="/patterns" className="btn-primary">
            View All Patterns
          </Link>
        </div>
      </div>
    </section>
  )
}
